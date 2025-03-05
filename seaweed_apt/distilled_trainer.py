from logger import logger,debug_memory 
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
import gc

import sys
# import os
# os.environ["FORCE_COLOR"] = "true"


torch.cuda.set_per_process_memory_fraction(0.8) 

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
    use_gradient_checkpointing=True  # for 24GB GPU
):
    """
    Train a consistency-distilled model from the original Wan model using precomputed contexts.

    Args:
        original_model: The original pre-trained Wan model
        config: Model configuration
        train_dataloader: DataLoader with precomputed dummy_data, positive_contexts, and negative_context
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save distilled model
        device: Training device (e.g., 'cuda')
        accelerator: Accelerator instance
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (paper uses 5e-6 for images)
        cfg_scale: Classifier-free guidance scale (paper uses 7.5)
        save_interval: Interval to save checkpoints (paper uses 350 updates)
        use_wandb: Whether to use Weights & Biases for logging
        project_name: WandB project name
        run_name: WandB run name
        t5_fsdp: Whether to use FSDP for T5 (unused here as text encoding is precomputed)
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency

    Returns:
        distilled_model: The trained consistency-distilled model (EMA version)
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
                "use_gradient_checkpointing": use_gradient_checkpointing,
            }
        )

    # Initialize distilled model from scratch with same architecture
    distilled_model = WanModel.from_pretrained(checkpoint_dir)

    # Create optimizer (paper uses RMSProp with alpha=0.9, equivalent to Adam beta2=0.9)
    optimizer = optim.RMSprop(distilled_model.parameters(), lr=learning_rate, alpha=0.9)

    # Set up accelerator (prepares distilled model, optimizer, and dataloader)
    distilled_model, optimizer, train_dataloader = accelerator.prepare(
        distilled_model, optimizer, train_dataloader
    )

    # Set both models to appropriate modes
    original_model.eval()
    distilled_model.train()

    # EMA setup (paper uses decay rate of 0.995)
    logger.debug("Setting up EMA model...")
    ema_model = WanModel.from_pretrained(checkpoint_dir, use_checkpoint=use_gradient_checkpointing)
    ema_model.eval()
    ema_decay = 0.995

    def update_ema(target_model, source_model, decay):
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

    # Initialize stats
    total_loss = 0.0
    step = 0

    # Ensure models start on CPU to free VRAM initially
    original_model = original_model.to('cpu')
    logger.debug("Initialized models on CPU")

    # Enhanced training loop with precomputed contexts
    for epoch in range(num_epochs):
        logger.debug(f"Starting epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            # Unpack batch: samples, positive_contexts, negative_contexts
            samples, positive_contexts, negative_contexts = batch
            logger.debug(f"Batch {batch_idx}: samples shape={samples.shape}, "
                         f"positive_contexts shape={positive_contexts.shape}, "
                         f"negative_contexts shape={negative_contexts.shape}")

            # Move samples and contexts to device
            samples = samples.to(device)
            context = positive_contexts.to(device)  # Already precomputed
            context_null = negative_contexts.to(device)  # Already precomputed

            # Generate random noise
            logger.debug("Generating random noise")
            noise = torch.randn_like(samples)  # Shape: [B, 16, T, H, W]
            noise = noise.squeeze(0) if noise.dim() == 5 else noise  # Adjust if squeezed earlier
            logger.debug(f"Adjusted noise shape: {noise.shape}")

            # Compute seq_len dynamically based on samples
            patch_size = original_model.patch_size  # (1, 2, 2)
            seq_len = (samples.shape[2] // patch_size[0]) * \
                      (samples.shape[3] // patch_size[1]) * \
                      (samples.shape[4] // patch_size[2])  # e.g., 1560 for [16, 1, 60, 104]
            logger.debug(f"Computed seq_len for batch: {seq_len}")

            timestep = torch.ones(samples.shape[0], device=device) * config.num_train_timesteps
            logger.debug(f"Timestep shape: {timestep.shape}, value: {timestep[0].item()}")

            # Move original_model to GPU for teacher prediction
            logger.debug(f"Moving original_model to {device}")
            original_model = original_model.to(device)
            torch.cuda.empty_cache()

            # Teacher prediction with CFG using precomputed contexts
            logger.debug("Starting teacher model prediction")
            with torch.no_grad():
                logger.debug("Running unconditional prediction with original model")
                v_uncond = original_model([noise], t=timestep, context=context_null, seq_len=seq_len)[0]
                logger.debug(f"Unconditional prediction shape: {v_uncond.shape}")

                logger.debug("Running conditional prediction with original model")
                v_cond = original_model([noise], t=timestep, context=context, seq_len=seq_len)[0]
                logger.debug(f"Conditional prediction shape: {v_cond.shape}")

                logger.debug(f"Applying classifier-free guidance with scale: {cfg_scale}")
                v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
                logger.debug(f"Teacher prediction shape: {v_teacher.shape}")

            # Move original_model back to CPU
            logger.debug("Moving original_model back to CPU")
            original_model = original_model.to('cpu')
            torch.cuda.empty_cache()

            # Student prediction
            logger.debug("Running student model prediction")
            v_student = distilled_model([noise], t=timestep, context=context, seq_len=seq_len)[0]
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
            distilled_model.to(device)  # Already on device typically
            ema_model.to(device)
            update_ema(ema_model, distilled_model, ema_decay)
            ema_model.to('cpu')  # Move back to CPU immediately
            torch.cuda.empty_cache()

            logger.debug(f"Before clean up - GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")
            logger.debug(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB")

            logger.debug("Cleaning up GPU memory")
            del v_uncond, v_cond, v_teacher, v_student, noise, context, context_null
            torch.cuda.empty_cache()

            # Ensure no lingering references
            samples = samples.cpu()
            optimizer.zero_grad(set_to_none=True)  # Clear gradients more aggressively

            logger.debug(f"After - GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")
            logger.debug(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB")

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

            # Save checkpoint with EMA weights
            if step % save_interval == 0 and accelerator.is_main_process:
                checkpoint_path = f"{output_dir}/consistency_model_step_{step}.pt"
                logger.debug(f"Saving checkpoint to {checkpoint_path}")
                unwrapped_ema = accelerator.unwrap_model(ema_model)
                torch.save(unwrapped_ema.state_dict(), checkpoint_path)
                logger.debug(f"Saved EMA checkpoint to {checkpoint_path}")

        # Save epoch checkpoint with EMA weights
        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/consistency_model_epoch_{epoch+1}.pt"
            logger.debug(f"Saving epoch checkpoint to {checkpoint_path}")
            unwrapped_ema = accelerator.unwrap_model(ema_model)
            torch.save(unwrapped_ema.state_dict(), checkpoint_path)
            logger.debug(f"Saved EMA epoch checkpoint to {checkpoint_path}")

        logger.debug(f"Completed epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

    # Save final EMA model
    if accelerator.is_main_process:
        final_path = f"{output_dir}/consistency_model_final.pt"
        logger.debug(f"Saving final EMA model to {final_path}")
        unwrapped_ema = accelerator.unwrap_model(ema_model)
        torch.save(unwrapped_ema.state_dict(), final_path)
        logger.debug(f"Saved final EMA consistency model to {final_path}")

    if use_wandb and accelerator.is_main_process:
        wandb.finish()

    return accelerator.unwrap_model(ema_model)  # Return EMA model as per paper


# Adjusted Dataset and Main Section
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load precomputed data including positive and negative contexts
        data_dict = torch.load(data_path, map_location='cpu')
        self.samples = data_dict['dummy_data']  # Video latents
        self.positive_contexts = data_dict['positive_contexts']  # Precomputed positive contexts
        self.negative_context = data_dict['negative_context']  # Precomputed negative context
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return sample and its corresponding precomputed contexts
        sample = self.samples[idx]
        positive_context = self.positive_contexts[idx]
        # Expand negative_context to match batch size (assuming it's a single tensor)
        negative_context = self.negative_context.expand_as(positive_context)
        return sample, positive_context, negative_context

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
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
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
    config = t2v_1_3B  # Use 1.3B here for simplicity; switch to t2v_14B for 14B scale
    if not hasattr(config, 'seq_len'):
        config.seq_len = 512  # Match the text encoder's sequence length

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

    # Check for generated data file with precomputed contexts
    data_file = "dummy_data_480x832.pt"  # Adjust size_str as per your generate_batch output
    if not os.path.exists(data_file):
        logger.error(f"Required file {data_file} not found.")
        logger.info("Please run generate_batch to create the dummy data with contexts first.")
        sys.exit(1)

    train_dataset = TextVideoDataset(data_file)

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