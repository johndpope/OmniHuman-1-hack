import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator
import copy
from model import *
from wan.modules.model import WanModel
from wan.text2video import WanT2V
from wan.utils.utils import str2bool

def train_seaweed_apt(config, 
                      original_model,
                      distilled_model, 
                      train_dataloader_image, 
                      train_dataloader_video,
                      device,
                      checkpoint_dir,
                      accelerator,
                      use_wandb=False,
                      project_name="seaweed-apt-training",
                      run_name=None,
                      use_checkpoint=True): 
    """
    Train Seaweed APT with Wan as the base model using Accelerate
    
    Args:
        config: Training configuration
        original_model: Original pre-trained Wan model
        distilled_model: Consistency-distilled Wan model
        train_dataloader_image: DataLoader for image training
        train_dataloader_video: DataLoader for video training
        device: Training device
        checkpoint_dir: Directory for saving checkpoints
        accelerator: Accelerator instance
        use_wandb: Whether to use Weights & Biases logging
        project_name: WandB project name
        run_name: WandB run name
    """
    # Create output directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "g_lr_image": config.g_lr_image,
                "d_lr_image": config.d_lr_image,
                "g_lr_video": config.g_lr_video,
                "d_lr_video": config.d_lr_video,
                "image_updates": config.image_updates,
                "video_updates": config.video_updates,
                "ema_decay": config.ema_decay,
                "lambda_r1": config.lambda_r1,
            }
        )
    
    # Initialize generator and discriminator
    generator = WanAPTGenerator(distilled_model, 
                               final_timestep=config.num_train_timesteps,
                               use_checkpoint=use_checkpoint)
    discriminator = WanAPTDiscriminator(original_model,
                                       use_checkpoint=use_checkpoint)
    
    # Setup optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), 
                                lr=config.g_lr_image, 
                                alpha=0.9)
    d_optimizer = optim.RMSprop(discriminator.parameters(), 
                                lr=config.d_lr_image, 
                                alpha=0.9)
    
    # Prepare models, optimizers and dataloaders with accelerator
    generator, discriminator, g_optimizer, d_optimizer, train_dataloader_image = accelerator.prepare(
        generator, discriminator, g_optimizer, d_optimizer, train_dataloader_image
    )
    
    # Create T5 text encoder for processing text prompts
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=f"{checkpoint_dir}/{config.t5_checkpoint}",
        tokenizer_path=f"{checkpoint_dir}/{config.t5_tokenizer}"
    )
    
    # Setup EMA
    ema = EMA(accelerator.unwrap_model(generator), decay=config.ema_decay)
    
    # Phase 1: Image Training
    if accelerator.is_main_process:
        print("Starting image training phase...")
    
    # Training loop for image phase
    for update in range(config.image_updates):
        for batch_idx, (images, text_prompts) in enumerate(train_dataloader_image):
            # Process text prompts through T5 encoder
            context = text_encoder(text_prompts, device)
            
            # Sample timestep with shift function
            t = torch.rand(images.shape[0], device=device) * config.num_train_timesteps
            s = 1.0  # For images
            t_shifted = s * t / (1.0 + (s - 1.0) * t)
            
            # Train discriminator
            # Generate random noise
            noise = torch.randn_like(images)
            
            # Generate fake samples
            with torch.no_grad():
                fake_images = generator(noise, context, config.seq_len)
            
            # Discriminator predictions
            real_logits = discriminator(images, t_shifted, context, config.seq_len)
            fake_logits = discriminator(fake_images, t_shifted, context, config.seq_len)
            
            # Compute discriminator loss
            d_loss = -torch.mean(torch.log(torch.sigmoid(real_logits) + 1e-8)) \
                      -torch.mean(torch.log(1 - torch.sigmoid(fake_logits) + 1e-8))
            
            # Add approximated R1 regularization
            r1_loss = approximated_r1_loss(
                discriminator, images, t_shifted, context, config.seq_len, sigma=0.01)
            d_loss += config.lambda_r1 * r1_loss
            
            # Update discriminator
            d_optimizer.zero_grad()
            accelerator.backward(d_loss)
            d_optimizer.step()
            
            # Train generator
            # Generate fake samples
            fake_images = generator(noise, context, config.seq_len)
            
            # Discriminator predictions on new fake samples
            fake_logits = discriminator(fake_images, t_shifted, context, config.seq_len)
            
            # Compute generator loss
            g_loss = -torch.mean(torch.log(torch.sigmoid(fake_logits) + 1e-8))
            
            # Update generator
            g_optimizer.zero_grad()
            accelerator.backward(g_loss)
            g_optimizer.step()
            
            # Update EMA model (on unwrapped model)
            if accelerator.sync_gradients:
                with accelerator.main_process_first():
                    ema.update(accelerator.unwrap_model(generator))
            
            # Log metrics
            if use_wandb and accelerator.is_main_process and batch_idx % 5 == 0:
                wandb.log({
                    "image_batch_idx": batch_idx,
                    "image_update": update,
                    "image_g_loss": g_loss.item(),
                    "image_d_loss": d_loss.item(),
                    "image_r1_loss": r1_loss.item(),
                })
            
            if accelerator.is_main_process and batch_idx % 10 == 0:
                print(f"[Image] Update {update}, Batch {batch_idx}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Save checkpoint after each update
        if accelerator.is_main_process and (update % 50 == 0 or update == config.image_updates - 1):
            unwrapped_generator = accelerator.unwrap_model(generator)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            
            torch.save({
                'generator': unwrapped_generator.state_dict(),
                'ema_generator': ema.model.state_dict(),
                'discriminator': unwrapped_discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'update': update,
            }, f"{checkpoint_dir}/seaweed_apt_image_update_{update}.pt")
    
    # Save final image phase checkpoint
    if accelerator.is_main_process:
        unwrapped_generator = accelerator.unwrap_model(generator)
        unwrapped_discriminator = accelerator.unwrap_model(discriminator)
        
        torch.save({
            'generator': unwrapped_generator.state_dict(),
            'ema_generator': ema.model.state_dict(),
            'discriminator': unwrapped_discriminator.state_dict(),
        }, f"{checkpoint_dir}/seaweed_apt_image_final.pt")
        
        print("Image phase complete. Starting video training phase...")
    
    # Wait for all processes to reach this point
    accelerator.wait_for_everyone()
    
    # Phase 2: Video Training
    # Re-initialize models for video phase
    
    # Load EMA generator from image phase
    if accelerator.is_main_process:
        checkpoint = torch.load(f"{checkpoint_dir}/seaweed_apt_image_final.pt")
        # Save the EMA weights temporarily to be loaded on all processes
        torch.save(checkpoint['ema_generator'], f"{checkpoint_dir}/temp_ema_weights.pt")
    
    # Wait for main process to save the weights
    accelerator.wait_for_everyone()
    
    # Initialize new models for video phase
    generator = WanAPTGenerator(distilled_model, final_timestep=config.num_train_timesteps)
    discriminator = WanAPTDiscriminator(original_model)
    
    # Load EMA weights into generator on all processes
    ema_weights = torch.load(f"{checkpoint_dir}/temp_ema_weights.pt")
    generator.load_state_dict(ema_weights)
    
    # Setup new optimizers with lower learning rate
    g_optimizer = optim.RMSprop(generator.parameters(), 
                               lr=config.g_lr_video, 
                               alpha=0.9)
    d_optimizer = optim.RMSprop(discriminator.parameters(), 
                               lr=config.d_lr_video, 
                               alpha=0.9)
    
    # Prepare models, optimizers and dataloaders with accelerator
    generator, discriminator, g_optimizer, d_optimizer, train_dataloader_video = accelerator.prepare(
        generator, discriminator, g_optimizer, d_optimizer, train_dataloader_video
    )
    
    # Setup EMA for video phase
    ema = EMA(accelerator.unwrap_model(generator), decay=config.ema_decay)
    
    # Training loop for video phase
    for update in range(config.video_updates):
        for batch_idx, (videos, text_prompts) in enumerate(train_dataloader_video):
            # Process text prompts through T5 encoder
            context = text_encoder(text_prompts, device)
            
            # Sample timestep with shift function
            t = torch.rand(videos.shape[0], device=device) * config.num_train_timesteps
            s = 12.0  # For videos
            t_shifted = s * t / (1.0 + (s - 1.0) * t)
            
            # Train discriminator
            # Generate random noise
            noise = torch.randn_like(videos)
            
            # Generate fake samples
            with torch.no_grad():
                fake_videos = generator(noise, context, config.seq_len)
            
            # Discriminator predictions
            real_logits = discriminator(videos, t_shifted, context, config.seq_len)
            fake_logits = discriminator(fake_videos, t_shifted, context, config.seq_len)
            
            # Compute discriminator loss
            d_loss = -torch.mean(torch.log(torch.sigmoid(real_logits) + 1e-8)) \
                      -torch.mean(torch.log(1 - torch.sigmoid(fake_logits) + 1e-8))
            
            # Add approximated R1 regularization (with higher sigma for videos)
            r1_loss = approximated_r1_loss(
                discriminator, videos, t_shifted, context, config.seq_len, sigma=0.1)
            d_loss += config.lambda_r1 * r1_loss
            
            # Update discriminator
            d_optimizer.zero_grad()
            accelerator.backward(d_loss)
            d_optimizer.step()
            
            # Train generator
            # Generate fake samples
            fake_videos = generator(noise, context, config.seq_len)
            
            # Discriminator predictions on new fake samples
            fake_logits = discriminator(fake_videos, t_shifted, context, config.seq_len)
            
            # Compute generator loss
            g_loss = -torch.mean(torch.log(torch.sigmoid(fake_logits) + 1e-8))
            
            # Update generator
            g_optimizer.zero_grad()
            accelerator.backward(g_loss)
            g_optimizer.step()
            
            # Update EMA model (on unwrapped model)
            if accelerator.sync_gradients:
                with accelerator.main_process_first():
                    ema.update(accelerator.unwrap_model(generator))
            
            # Log metrics
            if use_wandb and accelerator.is_main_process and batch_idx % 5 == 0:
                wandb.log({
                    "video_batch_idx": batch_idx,
                    "video_update": update,
                    "video_g_loss": g_loss.item(),
                    "video_d_loss": d_loss.item(),
                    "video_r1_loss": r1_loss.item(),
                })
            
            if accelerator.is_main_process and batch_idx % 10 == 0:
                print(f"[Video] Update {update}, Batch {batch_idx}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Save checkpoint after each update
        if accelerator.is_main_process and (update % 50 == 0 or update == config.video_updates - 1):
            unwrapped_generator = accelerator.unwrap_model(generator)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            
            torch.save({
                'generator': unwrapped_generator.state_dict(),
                'ema_generator': ema.model.state_dict(),
                'discriminator': unwrapped_discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'update': update,
            }, f"{checkpoint_dir}/seaweed_apt_video_update_{update}.pt")
    
    # Save final video phase checkpoint
    if accelerator.is_main_process:
        unwrapped_generator = accelerator.unwrap_model(generator)
        unwrapped_discriminator = accelerator.unwrap_model(discriminator)
        
        torch.save({
            'generator': unwrapped_generator.state_dict(),
            'ema_generator': ema.model.state_dict(),
            'discriminator': unwrapped_discriminator.state_dict(),
        }, f"{checkpoint_dir}/seaweed_apt_video_final.pt")
        
        # Clean up temporary files
        if os.path.exists(f"{checkpoint_dir}/temp_ema_weights.pt"):
            os.remove(f"{checkpoint_dir}/temp_ema_weights.pt")
        
        # Close wandb
        if use_wandb:
            wandb.finish()
    
    # Return the EMA model
    return ema.model


# Configuration class for training
class SeaweedAPTConfig:
    def __init__(self):
        # Model configuration
        self.num_train_timesteps = 1000
        self.text_len = 512
        self.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
        self.t5_tokenizer = "google/umt5-xxl"
        self.t5_dtype = torch.bfloat16
        self.seq_len = 1024  # Maximum sequence length
        
        # Training configuration
        self.image_batch_size = 9062
        self.video_batch_size = 2048
        self.g_lr_image = 5e-6
        self.d_lr_image = 5e-6
        self.g_lr_video = 3e-6
        self.d_lr_video = 3e-6
        self.image_updates = 350
        self.video_updates = 300
        self.ema_decay = 0.995
        self.lambda_r1 = 100.0
        
        # Image configuration
        self.image_resolution = 1024
        
        # Video configuration
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = 24
        self.video_frames = 48  # 2 seconds at 24fps


# Example usage:
if __name__ == "__main__":
    import argparse
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from wan.configs import t2v_14B, SIZE_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import str2bool
    
    parser = argparse.ArgumentParser(description="Train Seaweed APT with Wan base model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for saving checkpoints")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID (for original model)")
    parser.add_argument("--consistency_path", type=str, default="", help="Path to pretrained consistency model (optional)")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Whether to use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="seaweed-apt-stage2", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--config_file", type=str, default=None, help="Path to OmegaConf config file")
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config_file and os.path.exists(args.config_file):
        config_from_file = OmegaConf.load(args.config_file)
        # Create config object
        config = SeaweedAPTConfig()
        # Update config with values from file
        for key, value in config_from_file.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = SeaweedAPTConfig()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Set device - still needed for original model and other components
    device = accelerator.device
    
    # Initialize original Wan model (T2V-14B) - not wrapped by accelerator
    original_model = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        rank=0,
    ).model
    
    # Move original model to accelerator device
    original_model.to(device)
    
    # Load or train consistency-distilled model
    if args.consistency_path:
        if accelerator.is_main_process:
            print(f"Loading pre-trained consistency model from {args.consistency_path}")
        distilled_model = WanModel.from_pretrained(args.consistency_path)
    else:
        if accelerator.is_main_process:
            print("Training consistency model from scratch...")
        # Here you would implement the consistency distillation training
        # For now, we'll just use a copy of the original model as a placeholder
        distilled_model = copy.deepcopy(original_model)
    
    # Move distilled model to accelerator device
    distilled_model.to(device)
    
    # Setup data loaders (simplified for example)
    # In practice, you'd need proper dataset classes for images and videos
    from torch.utils.data import TensorDataset, DataLoader
    
    # Dummy data for example
    dummy_image_data = torch.randn(100, 16, 1, 128, 128)  # [N, C, T, H, W]
    dummy_image_prompts = ["A beautiful landscape"] * 100
    dummy_video_data = torch.randn(100, 16, 48, 90, 160)  # [N, C, T, H, W]
    dummy_video_prompts = ["A beautiful landscape in motion"] * 100
    
    train_dataset_image = TensorDataset(dummy_image_data, dummy_image_prompts)
    train_dataset_video = TensorDataset(dummy_video_data, dummy_video_prompts)
    
    train_dataloader_image = DataLoader(
        train_dataset_image, 
        batch_size=min(8, config.image_batch_size),  # Adjust based on memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    train_dataloader_video = DataLoader(
        train_dataset_video, 
        batch_size=min(4, config.video_batch_size),  # Adjust based on memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Start Seaweed APT training
    final_model = train_seaweed_apt(
        config=config,
        original_model=original_model,
        distilled_model=distilled_model,
        train_dataloader_image=train_dataloader_image,
        train_dataloader_video=train_dataloader_video,
        device=device,
        checkpoint_dir=args.output_dir,
        accelerator=accelerator,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
    )
    
    # Save final model
    if accelerator.is_main_process:
        torch.save(final_model.state_dict(), f"{args.output_dir}/seaweed_wan_apt_final.pt")
        print(f"Training complete! Final model saved to {args.output_dir}/seaweed_wan_apt_final.pt")