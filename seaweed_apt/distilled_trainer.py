from logger import logger,log_tensor_sizes, log_gpu_memory_usage,debug_memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator

from wan.modules.model import WanModel
from wan.utils.utils import str2bool
import gc
import sys
import torch.random
from torch.cuda.amp import GradScaler, autocast
torch.cuda.set_per_process_memory_fraction(0.8)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_consistency_distillation(
    config,
    train_dataloader,
    checkpoint_dir,
    output_dir,
    device,
    accelerator,
    num_epochs=10,
    learning_rate=5e-6,
    cfg_scale=7.5,
    save_interval=350,
    use_wandb=False,
    project_name="wan-consistency-distillation",
    run_name=None,
    use_gradient_checkpointing=True,
    gradient_accumulation_steps=16
):
    logger.debug("Initializing consistency distillation training...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # WandB setup
    if use_wandb and accelerator.is_main_process:
        wandb.init(project=project_name, name=run_name, config={
            "learning_rate": learning_rate, 
            "num_epochs": num_epochs, 
            "save_interval": save_interval,
            "method": "consistency_distillation", 
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "gradient_accumulation_steps": gradient_accumulation_steps
        })

    # Model setup
    logger.debug(f"Loading model from {checkpoint_dir}")
    distilled_model = HiDETRWanModel.from_pretrained(checkpoint_dir)
    distilled_model.use_checkpoint = use_gradient_checkpointing
    
    # Use AdamW optimizer for lower memory usage
    logger.debug("Initializing AdamW optimizer")
    optimizer = optim.AdamW(
        distilled_model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Accelerator preparation
    logger.debug("Preparing model and optimizer with accelerator")
    distilled_model, optimizer, train_dataloader = accelerator.prepare(
        distilled_model, optimizer, train_dataloader
    )

    distilled_model.train()
    
    # Initialize EMA model on CPU
    logger.debug("Initializing EMA model on CPU")
    ema_model = WanModel.from_pretrained(checkpoint_dir)
    ema_model.eval()
    ema_model.requires_grad_(False)
    ema_model = ema_model.cpu()
    ema_decay = 0.995

    # Mixed precision setup
    logger.debug("Setting up GradScaler for mixed precision")
    scaler = GradScaler()
    
    # Try to register objects for checkpointing if the method exists
    if hasattr(accelerator, "register_for_checkpointing"):
        accelerator.register_for_checkpointing(scaler)

    # Training loop variables
    total_loss = 0.0
    global_step = 0
    samples_processed = 0

    # Main training loop
    for epoch in range(num_epochs):
        logger.debug(f"Starting epoch {epoch+1}/{num_epochs}")
        debug_memory(f"Before epoch {epoch+1}")
        epoch_loss = 0.0
        
        # Reset optimizer at the start of each epoch
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            # Determine if this is the last step in accumulation cycle
            is_last_accumulation_step = (
                (batch_idx + 1) % gradient_accumulation_steps == 0 or 
                batch_idx == len(train_dataloader) - 1
            )
            
            # Log batch information
            debug_memory(f"Before batch {batch_idx}")
            
            # Execute the training step
            step_output = training_step(
                batch=batch,
                distilled_model=distilled_model,
                config=config,
                device=device,
                scaler=scaler,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
                is_last_step=is_last_accumulation_step
            )
            
            # Track losses
            batch_loss = step_output["loss"]
            total_loss += batch_loss
            epoch_loss += batch_loss
            samples_processed += batch[0].size(0)
            
            # After successful accumulation cycle
            if step_output["is_accumulation_complete"]:
                # Update global step counter
                global_step += 1
                
                # Update EMA model
                unwrapped_model = accelerator.unwrap_model(distilled_model)
                update_ema_model(ema_model, unwrapped_model, ema_decay)
                
                # Save checkpoint periodically
                if global_step % save_interval == 0:
                    # Create checkpoint directory
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint_{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    logger.debug(f"Saving checkpoint at step {global_step}")
                    # Use save_state or manually save depending on what's available
                    if hasattr(accelerator, "save_state"):
                        accelerator.save_state(checkpoint_dir)
                    else:
                        # Manual save as fallback - save unwrapped model
                        if accelerator.is_main_process:
                            unwrapped_model = accelerator.unwrap_model(distilled_model)
                            torch.save({
                                'model': unwrapped_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scaler': scaler.state_dict() if scaler is not None else None,
                                'step': global_step,
                                'epoch': epoch,
                            }, os.path.join(checkpoint_dir, "pytorch_model.bin"))
                    
                    # Also save EMA model separately
                    if accelerator.is_main_process:
                        ema_checkpoint_path = f"{output_dir}/ema_model_step_{global_step}.pt"
                        logger.debug(f"Saving EMA model to {ema_checkpoint_path}")
                        torch.save(ema_model.state_dict(), ema_checkpoint_path)
                
                # Log metrics
                if use_wandb and accelerator.is_main_process and global_step % 5 == 0:
                    avg_loss = total_loss / (samples_processed or 1)
                    wandb.log({
                        "step": global_step, 
                        "batch_loss": batch_loss,
                        "avg_loss": avg_loss,
                        "epoch": epoch + 1,
                        "samples_processed": samples_processed
                    })
                
                # Reset accumulation variables if needed
                debug_memory(f"After accumulation step {global_step}")
            
        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.debug(f"Epoch {epoch+1} completed with average loss: {avg_epoch_loss}")
        
        # Save epoch checkpoint 
        epoch_checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(epoch_checkpoint_dir, exist_ok=True)
        
        logger.debug(f"Saving checkpoint for epoch {epoch+1}")
        if hasattr(accelerator, "save_state"):
            accelerator.save_state(epoch_checkpoint_dir)
        else:
            # Manual save as fallback
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(distilled_model)
                torch.save({
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'step': global_step,
                    'epoch': epoch,
                }, os.path.join(epoch_checkpoint_dir, "pytorch_model.bin"))
        
        # Save EMA model for this epoch
        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/ema_model_epoch_{epoch+1}.pt"
            logger.debug(f"Saving epoch EMA checkpoint to {checkpoint_path}")
            torch.save(ema_model.state_dict(), checkpoint_path)
            logger.debug(f"Epoch EMA checkpoint saved")
            
            if use_wandb:
                wandb.log({"epoch": epoch + 1, "epoch_loss": avg_epoch_loss})

    # Save final model
    if accelerator.is_main_process:
        final_path = f"{output_dir}/ema_model_final.pt"
        logger.debug(f"Saving final EMA model to {final_path}")
        torch.save(ema_model.state_dict(), final_path)
        logger.debug("Final EMA model saved")

    # Clean up
    if use_wandb and accelerator.is_main_process:
        wandb.finish()
        
    logger.debug("Training completed successfully")
    return ema_model

def training_step(batch, distilled_model, config, device, scaler, optimizer, gradient_accumulation_steps, is_last_step=False):
    # Extract and move data to device
    noise, positive_contexts, v_teacher = batch
    noise = noise.to(device)
    context = positive_contexts.to(device)
    v_teacher = v_teacher.to(device)
    
    log_tensor_sizes({
        "noise": noise, 
        "positive_contexts": positive_contexts, 
        "v_teacher": v_teacher
    }, "Before moving to GPU")

    # Check tensor integrity
    logger.debug(f"noise shape: {noise.shape}, dtype: {noise.dtype}")
    logger.debug(f"context shape: {context.shape}, dtype: {context.dtype}")
    logger.debug(f"v_teacher shape: {v_teacher.shape}, dtype: {v_teacher.dtype}")

    # Prepare model inputs
    contexts_list = [context[i] for i in range(context.size(0))]
    patch_size = distilled_model.patch_size
    seq_len = (noise.shape[2] // patch_size[0]) * \
              (noise.shape[3] // patch_size[1]) * \
              (noise.shape[4] // patch_size[2])
    timestep = torch.ones(noise.shape[0], device=device) * config.num_train_timesteps
    
    # Forward pass with mixed precision
    with autocast():
        logger.debug("Forward pass: model prediction")
        
        # Try-except to capture errors
        try:
            v_student_output = distilled_model(
                noise, 
                t=timestep, 
                context=contexts_list,
                seq_len=seq_len
            )
            
            # Check if output is valid
            if v_student_output is None or len(v_student_output) == 0:
                logger.error("Model output is None or empty")
                raise ValueError("Model returned None or empty output")
                
            v_student = v_student_output[0]
            logger.debug(f"v_student shape: {v_student.shape}, dtype: {v_student.dtype}")
            
            # Scale loss for accumulation
            loss = F.mse_loss(v_student, v_teacher) / gradient_accumulation_steps
            logger.debug(f"Calculated loss: {loss.item() * gradient_accumulation_steps}")
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

    # Immediate cleanup of unused tensors
    del contexts_list, timestep
    torch.cuda.empty_cache()
    
    # Backward pass with memory management
    logger.debug("Backward pass: calculating gradients")
    scaler.scale(loss).backward()
    
    # Capture loss value and cleanup remaining tensors
    loss_value = loss.item() * gradient_accumulation_steps
    del v_student_output, v_student, loss
    torch.cuda.empty_cache()
    
    # Only keep gradients, remove everything else
    del noise, context, v_teacher
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "loss": loss_value,
        "is_accumulation_complete": is_last_step,
    }


def update_ema_model(ema_model, model, decay):
    """Updates EMA model parameters with CPU operations to save memory
    
    Args:
        ema_model: The EMA model (kept on CPU)
        model: The current training model
        decay: EMA decay factor
    """
    logger.debug("Updating EMA model")
    with torch.no_grad():
        for target_param, source_param in zip(ema_model.parameters(), model.parameters()):
            # Move source parameter to CPU for the update
            cpu_param = source_param.detach().cpu()
            target_param.data.mul_(decay).add_(cpu_param, alpha=1 - decay)
    logger.debug("EMA update completed")
    torch.cuda.empty_cache()

# Dataset (unchanged)
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data_dict = torch.load(data_path, map_location='cpu')
        self.noise = data_dict['noise']  # [100, 16, 1, 60, 104]
        self.positive_contexts = data_dict['positive_contexts']  # List of [512, 4096]
        self.v_teacher = data_dict['v_teacher']  # [100, 16, 1, 60, 104]
        self.num_samples = len(self.noise)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.noise[idx], self.positive_contexts[idx], self.v_teacher[idx]

if __name__ == "__main__":
    import argparse
    from wan.configs import t2v_14B, t2v_1_3B

    parser = argparse.ArgumentParser(description="Train consistency distillation for Seaweed-APT")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan T2V model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate (paper: 5e-6)")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale (paper: 7.5)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="seaweed-apt-distillation", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--config_file", type=str, default="./config.yaml", help="Path to config file")
    parser.add_argument("--save_interval", type=int, default=350, help="Save interval (paper: 350 updates)")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps for gradient accumulation")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    args = parser.parse_args()

    if args.config_file and os.path.exists(args.config_file):
        config = OmegaConf.load(args.config_file)
        args_dict = vars(args)
        for key, value in config.items():
            if key not in args_dict or args_dict[key] is None:
                args_dict[key] = value
        args = argparse.Namespace(**args_dict)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    config = t2v_1_3B
    if not hasattr(config, 'seq_len'):
        config.seq_len = 1560  # Computed based on 480x832 resolution

    data_file = "dummy_data_480x832.pt"
    if not os.path.exists(data_file):
        logger.error(f"Required file {data_file} not found.")
        logger.info("Please run generate_batch to create the dummy data with contexts first.")
        sys.exit(1)

    train_dataset = TextVideoDataset(data_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    distilled_model = train_consistency_distillation(
        config=config,
        train_dataloader=train_dataloader,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=device,
        accelerator=accelerator,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cfg_scale=args.cfg_scale,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )