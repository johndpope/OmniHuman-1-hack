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
from wan.text2video import WanT2V
from wan.utils.utils import str2bool
from logger import logger 


def train_consistency_distillation(
    original_model,
    config,
    train_dataloader,
    checkpoint_dir,
    output_dir,
    device,
    accelerator,
    num_epochs=10,
    learning_rate=5e-6,  # Aligned with paper's image training LR
    cfg_scale=7.5,       # Paper uses constant CFG scale of 7.5
    save_interval=350,   # Paper takes EMA checkpoint after 350 updates
    use_wandb=False,
    project_name="wan-consistency-distillation",
    run_name=None,
    t5_fsdp=False,
):
    """
    Train a consistency-distilled model from the original Wan model.
    
    Args:
        original_model: The original pre-trained Wan model
        config: Model configuration 
        train_dataloader: DataLoader for training data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save distilled model
        device: Training device
        accelerator: Accelerator instance
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (paper uses 5e-6 for images)
        cfg_scale: Classifier-free guidance scale (paper uses 7.5)
        save_interval: Interval to save checkpoints (paper uses 350 updates)
        use_wandb: Whether to use Weights & Biases for logging
        project_name: WandB project name
        run_name: WandB run name
    
    Returns:
        distilled_model: The trained consistency-distilled model
    """
    logger.debug("Initializing consistency distillation training...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "cfg_scale": cfg_scale,
                "save_interval": save_interval,
                "method": "consistency_distillation",
            }
        )
    
    # Initialize distilled model from scratch with same architecture
    distilled_model = WanModel.from_pretrained(checkpoint_dir)
    
    # Create optimizer (paper uses RMSProp with alpha=0.9, equivalent to Adam beta2=0.9)
    optimizer = optim.RMSprop(distilled_model.parameters(), lr=learning_rate, alpha=0.9)

    # Set up accelerator
    distilled_model, optimizer, train_dataloader = accelerator.prepare(
        distilled_model, optimizer, train_dataloader
    )
    
    # Set both models to appropriate modes
    original_model.eval()
    distilled_model.train()
    
    # Create T5 text encoder for processing text prompts

    from wan.modules.t5 import T5EncoderModel
    from wan.distributed.fsdp import shard_model
    from functools import partial
    shard_fn = partial(shard_model, device_id=0)
    t5_device = torch.device('cpu')
    logger.debug(f"Initializing text encoder on {t5_device}...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=t5_device, # T5EncoderModel is CPU-only unless you have a GPU with 24GB+ VRAM
        checkpoint_path=f"{checkpoint_dir}/{config.t5_checkpoint}",
        tokenizer_path=f"{checkpoint_dir}/{config.t5_tokenizer}",
        shard_fn=shard_fn if t5_fsdp else None,
    )
    # Negative prompt for CFG (paper uses fixed negative prompt)
    negative_prompt = config.sample_neg_prompt or ""  # Empty string if not specified
    # EMA setup (paper uses decay rate of 0.995)
    logger.debug("Setting up EMA model...")
    ema_model = WanModel.from_pretrained(checkpoint_dir)
    ema_model.eval()
    ema_decay = 0.995
    
    def update_ema(target_model, source_model, decay):
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)
    

    
    def process_text(prompts):
        """
        Process text prompts with detailed logging for debugging.
        """
        logger.debug(f"Processing text prompts: {prompts[:2]}{'...' if len(prompts) > 2 else ''}")
        logger.debug(f"Tokenizer type: {type(text_encoder.tokenizer)}")
        
        # Get token IDs and mask - explicitly request mask to be returned
        logger.debug("Calling tokenizer with return_mask=True")
        ids, mask = text_encoder.tokenizer(prompts, return_mask=True)
        logger.debug(f"Tokenizer returned: ids shape={ids.shape}, mask shape={mask.shape}")
        
        # Process on CPU (where text_encoder model is)
        with torch.no_grad():
            # Make sure tokens are on the same device as the model
            logger.debug(f"Moving tokens to device: {t5_device}")
            ids = ids.to(t5_device)
            mask = mask.to(t5_device)
            
            logger.debug(f"Running text encoder model with ids shape={ids.shape}, mask shape={mask.shape}")
            context = text_encoder.model(ids, mask)
            logger.debug(f"Text encoder output context shape: {context.shape}")
        
        # Move output to the target device for further processing
        logger.debug(f"Moving context to device: {device}")
        result = context.to(device)
        logger.debug(f"Final context shape: {result.shape}")
        return result

    # Enhanced training loop with detailed logging
    for epoch in range(num_epochs):
        logger.debug(f"Starting epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (samples, text_prompts) in enumerate(tqdm(train_dataloader)):
            logger.debug(f"Batch {batch_idx}: samples shape={samples.shape}, text_prompts length={len(text_prompts)}")
            logger.debug(f"Sample text prompt example: {text_prompts[0][:50]}...")
            
            # Move samples to device
            logger.debug(f"Moving samples to device: {device}")
            samples = samples.to(device)
            
            # Process text prompts
            logger.debug("Processing positive text prompts")
            context = process_text(text_prompts)
            logger.debug(f"Positive context shape: {context[0].shape if isinstance(context, list) else context.shape}")
            
            logger.debug("Processing negative text prompts")
            context_null = process_text([negative_prompt] * len(text_prompts))
            logger.debug(f"Negative context shape: {context_null[0].shape if isinstance(context_null, list) else context_null.shape}")
            
            # Generate random noise
            logger.debug("Generating random noise")
            noise = torch.randn_like(samples)
            logger.debug(f"Noise shape: {noise.shape}")
            
            # Final timestep for one-step prediction
            logger.debug("Creating timestep tensor")
            timestep = torch.ones(samples.shape[0], device=device) * config.num_train_timesteps
            logger.debug(f"Timestep shape: {timestep.shape}, value: {timestep[0].item()}")
            
            # Teacher prediction with CFG
            logger.debug("Starting teacher model prediction")
            with torch.no_grad():
                # Unconditional prediction
                logger.debug("Running unconditional prediction with original model")
                v_uncond = original_model(
                    [noise], 
                    t=timestep, 
                    context=context_null, 
                    seq_len=config.seq_len
                )[0]
                logger.debug(f"Unconditional prediction shape: {v_uncond.shape}")
                
                # Conditional prediction
                logger.debug("Running conditional prediction with original model")
                v_cond = original_model(
                    [noise], 
                    t=timestep, 
                    context=context, 
                    seq_len=config.seq_len
                )[0]
                logger.debug(f"Conditional prediction shape: {v_cond.shape}")
                
                # CFG: v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
                logger.debug(f"Applying classifier-free guidance with scale: {cfg_scale}")
                v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
                logger.debug(f"Teacher prediction shape: {v_teacher.shape}")
            
            # Student prediction
            logger.debug("Running student model prediction")
            v_student = distilled_model(
                [noise], 
                t=timestep, 
                context=context, 
                seq_len=config.seq_len
            )[0]
            logger.debug(f"Student prediction shape: {v_student.shape}")
            
            # MSE loss calculation
            logger.debug("Calculating MSE loss")
            loss = F.mse_loss(v_student, v_teacher)
            logger.debug(f"Loss value: {loss.item()}")
            
            # Backpropagation
            logger.debug("Starting backpropagation")
            accelerator.backward(loss)
            logger.debug("Completed backpropagation")
            
            logger.debug("Updating optimizer")
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            logger.debug("Updating EMA model")
            update_ema(ema_model, distilled_model, ema_decay)
            
            # Update stats
            total_loss += loss.item()
            step += 1
            
            # Log to wandb
            if use_wandb and accelerator.is_main_process and batch_idx % 5 == 0:
                logger.debug("Logging to wandb")
                wandb.log({
                    "step": step,
                    "batch_loss": loss.item(),
                    "avg_loss": total_loss / (batch_idx + 1),
                    "epoch": epoch + 1,
                })
            
            # Print progress
            if accelerator.is_main_process and batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.debug(f"Progress update: Epoch {epoch+1}, Batch {batch_idx}, Avg Loss: {avg_loss:.6f}")
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.6f}")
            
            # Save checkpoint with EMA weights
            if step % save_interval == 0 and accelerator.is_main_process:
                checkpoint_path = f"{output_dir}/consistency_model_step_{step}.pt"
                logger.debug(f"Saving checkpoint to {checkpoint_path}")
                unwrapped_ema = accelerator.unwrap_model(ema_model)
                torch.save(unwrapped_ema.state_dict(), checkpoint_path)
                logger.debug("Checkpoint saved successfully")
                logger.debug(f"Saved EMA checkpoint to {checkpoint_path}")
        
        # Save epoch checkpoint with EMA weights
        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/consistency_model_epoch_{epoch+1}.pt"
            logger.debug(f"Saving epoch checkpoint to {checkpoint_path}")
            unwrapped_ema = accelerator.unwrap_model(ema_model)
            torch.save(unwrapped_ema.state_dict(), checkpoint_path)
            logger.debug("Epoch checkpoint saved successfully")
            logger.debug(f"Saved EMA epoch checkpoint to {checkpoint_path}")
        
        logger.debug(f"Completed epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0
        logger.debug(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (samples, text_prompts) in enumerate(tqdm(train_dataloader)):
            logger.debug(f"Processing text prompts for batch {batch_idx}")

            # Process text prompts
            context = process_text(text_prompts)
            context_null = process_text([negative_prompt] * len(text_prompts))
            # Generate random noise
            noise = torch.randn_like(samples)
            
            # Final timestep for one-step prediction (paper uses T)
            timestep = torch.ones(samples.shape[0], device=device) * config.num_train_timesteps
            
            # Teacher prediction with CFG
            with torch.no_grad():
                # Unconditional prediction
                v_uncond = original_model(
                    [noise], 
                    t=timestep, 
                    context=context_null, 
                    seq_len=config.seq_len
                )[0]
                
                # Conditional prediction
                v_cond = original_model(
                    [noise], 
                    t=timestep, 
                    context=context, 
                    seq_len=config.seq_len
                )[0]
                
                # CFG: v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
                v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Student prediction
            v_student = distilled_model(
                [noise], 
                t=timestep, 
                context=context, 
                seq_len=config.seq_len
            )[0]
            
            # MSE loss (paper uses mean squared error for consistency distillation)
            loss = F.mse_loss(v_student, v_teacher)
            
            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            update_ema(ema_model, distilled_model, ema_decay)
            
            # Update stats
            total_loss += loss.item()
            step += 1
            
            # Log to wandb
            if use_wandb and accelerator.is_main_process and batch_idx % 5 == 0:
                wandb.log({
                    "step": step,
                    "batch_loss": loss.item(),
                    "avg_loss": total_loss / (batch_idx + 1),
                    "epoch": epoch + 1,
                })
            
            # Print progress
            if accelerator.is_main_process and batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.6f}")
            
            # Save checkpoint with EMA weights
            if step % save_interval == 0 and accelerator.is_main_process:
                checkpoint_path = f"{output_dir}/consistency_model_step_{step}.pt"
                unwrapped_ema = accelerator.unwrap_model(ema_model)
                torch.save(unwrapped_ema.state_dict(), checkpoint_path)
                logger.debug(f"Saved EMA checkpoint to {checkpoint_path}")
        
        # Save epoch checkpoint with EMA weights
        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/consistency_model_epoch_{epoch+1}.pt"
            unwrapped_ema = accelerator.unwrap_model(ema_model)
            torch.save(unwrapped_ema.state_dict(), checkpoint_path)
            logger.debug(f"Saved EMA epoch checkpoint to {checkpoint_path}")
        
        total_loss = 0.0
    
    # Save final EMA model
    if accelerator.is_main_process:
        final_path = f"{output_dir}/consistency_model_final.pt"
        unwrapped_ema = accelerator.unwrap_model(ema_model)
        torch.save(unwrapped_ema.state_dict(), final_path)
        logger.debug(f"Saved final EMA consistency model to {final_path}")
    
    if use_wandb and accelerator.is_main_process:
        wandb.finish()
    
    return accelerator.unwrap_model(ema_model)  # Return EMA model as per paper


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
    parser.add_argument(
    "--t5_cpu",
    action="store_true",
    default=False,
    help="Whether to place T5 model on CPU.")
    parser.add_argument(
    "--dit_fsdp",
    action="store_true",
    default=False,
    help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        config = OmegaConf.load(args.config_file)
        args_dict = vars(args)
        for key, value in config.items():
            if key not in args_dict or args_dict[key] is None:
                args_dict[key] = value
        args = argparse.Namespace(**args_dict)
    
    # Initialize accelerator with BF16 mixed precision (paper uses BF16)
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    # Initialize teacher model
    config = t2v_1_3B #t2v_14B  # Use 14B model to align with paper's 8B parameter scale
    original_model = WanT2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu
    ).model.to(device)
    

    class TextVideoDataset(torch.utils.data.Dataset):
        def __init__(self, video_tensors, text_prompts):
            """
            Custom dataset for text-to-video training
            
            Args:
                video_tensors: Tensor of video data with shape [N, C, T, H, W]
                text_prompts: List of text prompts (strings)
            """
            self.video_tensors = video_tensors
            self.text_prompts = text_prompts
            assert len(video_tensors) == len(text_prompts), "Number of videos and prompts must match"
            
        def __len__(self):
            return len(self.video_tensors)
        
        def __getitem__(self, idx):
            return self.video_tensors[idx], self.text_prompts[idx]

    # Create proper dataset with videos and text prompts
    dummy_data = torch.randn(100, 16, 1, 128, 128)  # [N, C, T, H, W]
    dummy_prompts = ["A beautiful landscape"] * 100
    train_dataset = TextVideoDataset(dummy_data, dummy_prompts)



    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    # Train
    distilled_model = train_consistency_distillation(
        original_model=original_model,
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
    )