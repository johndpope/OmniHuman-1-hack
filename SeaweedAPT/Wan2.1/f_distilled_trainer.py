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
import math

from wan.modules.model import WanModel
from wan.text2video import WanT2V
from wan.utils.utils import str2bool

# f-divergence helper functions
# https://arxiv.org/html/2502.15681v1  
# One-step Diffusion Models with f-Divergence Distribution Matching
def get_f_divergence_fn(divergence_type="reverse-kl"):
    """Return the appropriate weighting function based on f-divergence type."""
    if divergence_type == "reverse-kl":
        # h(r) = 1
        return lambda r: torch.ones_like(r)
    elif divergence_type == "forward-kl":
        # h(r) = r
        return lambda r: r
    elif divergence_type == "jensen-shannon":
        # h(r) = r/(r+1)
        return lambda r: r / (r + 1)
    elif divergence_type == "squared-hellinger":
        # h(r) = 1/(4*sqrt(r))
        return lambda r: 1 / (4 * torch.sqrt(r + 1e-8))
    elif divergence_type == "softened-rkl":
        # h(r) = 1/(r+1)
        return lambda r: 1 / (r + 1)
    else:
        raise ValueError(f"Unsupported f-divergence type: {divergence_type}")

def train_consistency_distillation(
    original_model,
    config,
    train_dataloader,
    checkpoint_dir,
    output_dir,
    device,
    accelerator,
    num_epochs=10,
    learning_rate=1e-5,
    cfg_scale=7.5,
    save_interval=10,
    use_wandb=False,
    project_name="wan-consistency-distillation",
    run_name=None,
    f_divergence="jensen-shannon",
    use_discriminator=True,
    discriminator_lr=1e-5,
    alpha=0.1,  # Weight for GAN loss
    beta=0.9,   # Weight for f-divergence loss
):
    """
    Train a consistency-distilled model from the original Wan model with f-divergence.
    
    Args:
        original_model: The original pre-trained Wan model
        config: Model configuration 
        train_dataloader: DataLoader for training data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save distilled model
        device: Training device
        accelerator: Accelerator instance
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        cfg_scale: Classifier-free guidance scale
        save_interval: Interval to save checkpoints (in steps)
        use_wandb: Whether to use Weights & Biases for logging
        project_name: WandB project name
        run_name: WandB run name
        f_divergence: Type of f-divergence to use ("reverse-kl", "forward-kl", "jensen-shannon", etc.)
        use_discriminator: Whether to use a discriminator for density ratio estimation
        discriminator_lr: Learning rate for discriminator
        alpha: Weight for GAN loss
        beta: Weight for f-divergence loss
    
    Returns:
        distilled_model: The trained consistency-distilled model
    """
    if accelerator.is_main_process:
        print(f"Initializing consistency distillation training with {f_divergence} divergence...")
    
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
                "f_divergence": f_divergence,
                "use_discriminator": use_discriminator,
                "discriminator_lr": discriminator_lr,
                "alpha": alpha,
                "beta": beta,
            }
        )
    
    # Initialize distilled model from scratch with same architecture
    distilled_model = WanModel.from_pretrained(checkpoint_dir)
    
    # Create a lightweight discriminator if needed for density ratio estimation
    discriminator = None
    d_optimizer = None
    if use_discriminator:
        # Simple discriminator that takes model features
        discriminator = nn.Sequential(
            nn.Linear(512, 256),  # Assume 512 is the feature dimension
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        ).to(device)
        d_optimizer = optim.AdamW(discriminator.parameters(), lr=discriminator_lr)
    
    # Create optimizer for the distilled model
    optimizer = optim.AdamW(distilled_model.parameters(), lr=learning_rate)
    
    # Prepare models, optimizers, and dataloaders with accelerator
    if use_discriminator:
        distilled_model, discriminator, optimizer, d_optimizer, train_dataloader = accelerator.prepare(
            distilled_model, discriminator, optimizer, d_optimizer, train_dataloader
        )
    else:
        distilled_model, optimizer, train_dataloader = accelerator.prepare(
            distilled_model, optimizer, train_dataloader
        )
    
    # Set models to appropriate modes
    original_model.eval()
    distilled_model.train()
    if use_discriminator:
        discriminator.train()
    
    # Create T5 text encoder for processing text prompts
    from wan.modules.t5 import T5EncoderModel
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=f"{checkpoint_dir}/{config.t5_checkpoint}",
        tokenizer_path=f"{checkpoint_dir}/{config.t5_tokenizer}"
    )
    
    # Negative prompt for classifier-free guidance
    negative_prompt = config.sample_neg_prompt
    
    # Get appropriate f-divergence weighting function
    f_weight_fn = get_f_divergence_fn(f_divergence)
    
    # Training loop
    step = 0
    total_loss = 0.0
    total_g_loss = 0.0
    total_d_loss = 0.0
    
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (samples, text_prompts) in enumerate(tqdm(train_dataloader)):
            # Process text prompts through text encoder
            context = text_encoder(text_prompts, device)
            context_null = text_encoder([negative_prompt] * len(text_prompts), device)
            
            # Generate random noise
            noise = torch.randn_like(samples)
            
            # Use final timestep for one-step prediction
            timestep = torch.ones(samples.shape[0], device=device) * config.num_train_timesteps
            
            # Compute teacher prediction with classifier-free guidance
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
                
                # Apply classifier-free guidance
                v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Compute student prediction
            v_student = distilled_model(
                [noise], 
                t=timestep, 
                context=context, 
                seq_len=config.seq_len
            )[0]
            
            # Extract features for discriminator
            # Assuming we can extract intermediate features from the models
            # This depends on the actual model architecture
            teacher_features = v_teacher.view(v_teacher.size(0), -1)[:, :512]  # Example feature extraction
            student_features = v_student.view(v_student.size(0), -1)[:, :512]
            
            # Discriminator step (if using)
            d_loss = torch.tensor(0.0, device=device)
            if use_discriminator:
                # Train discriminator
                if d_optimizer is not None:
                    d_optimizer.zero_grad()
                
                # Get discriminator predictions
                real_logits = discriminator(teacher_features.detach())
                fake_logits = discriminator(student_features.detach())
                
                # Compute discriminator loss (standard GAN loss)
                d_loss = -torch.mean(torch.log(torch.sigmoid(real_logits) + 1e-8)) \
                         -torch.mean(torch.log(1 - torch.sigmoid(fake_logits) + 1e-8))
                
                # Update discriminator
                accelerator.backward(d_loss)
                d_optimizer.step()
                
                # Get density ratio estimates from discriminator for weighting
                with torch.no_grad():
                    # For stabilized training, we clip the density ratio
                    density_ratio = torch.exp(discriminator(student_features))
                    density_ratio = torch.clamp(density_ratio, 0.01, 100.0)
            else:
                # Use a constant density ratio if no discriminator
                density_ratio = torch.ones_like(student_features[:, 0]).unsqueeze(1)
            
            # Get the score difference between teacher and student
            score_diff = v_teacher - v_student
            
            # Apply f-divergence weighting
            weights = f_weight_fn(density_ratio)
            
            # Normalize weights for stability if needed
            weights = weights / (weights.mean() + 1e-8)
            
            # Compute weighted loss
            # Expand weights to match dimensions of score_diff if needed
            weights_expanded = weights.view(-1, 1, 1, 1, 1)  # Adjust based on actual dimensions
            weighted_score_diff = weights_expanded * score_diff
            
            # Compute f-divergence loss
            f_div_loss = torch.mean(weighted_score_diff ** 2)
            
            # GAN loss for generator (if using discriminator)
            g_loss = torch.tensor(0.0, device=device)
            if use_discriminator:
                g_logits = discriminator(student_features)
                g_loss = -torch.mean(torch.log(torch.sigmoid(g_logits) + 1e-8))
            
            # Combined loss
            loss = beta * f_div_loss
            if use_discriminator:
                loss += alpha * g_loss
            
            # Update generator
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            if use_discriminator:
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
            step += 1
            
            # Log to wandb
            if use_wandb and accelerator.is_main_process and batch_idx % 5 == 0:
                log_dict = {
                    "step": step,
                    "batch_loss": loss.item(),
                    "f_div_loss": f_div_loss.item(),
                    "avg_loss": total_loss / (batch_idx + 1),
                    "epoch": epoch + 1,
                    "weight_mean": weights.mean().item(),
                    "weight_std": weights.std().item(),
                }
                if use_discriminator:
                    log_dict.update({
                        "g_loss": g_loss.item(),
                        "d_loss": d_loss.item(),
                        "avg_g_loss": total_g_loss / (batch_idx + 1),
                        "avg_d_loss": total_d_loss / (batch_idx + 1),
                        "density_ratio_mean": density_ratio.mean().item(),
                        "density_ratio_std": density_ratio.std().item(),
                    })
                wandb.log(log_dict)
            
            # Print progress
            if accelerator.is_main_process and batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                log_str = f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.6f}"
                if use_discriminator:
                    avg_g_loss = total_g_loss / (batch_idx + 1)
                    avg_d_loss = total_d_loss / (batch_idx + 1)
                    log_str += f", G Loss: {avg_g_loss:.6f}, D Loss: {avg_d_loss:.6f}"
                print(log_str)
            
            # Save checkpoint
            if step % save_interval == 0 and accelerator.is_main_process:
                checkpoint_path = f"{output_dir}/consistency_model_step_{step}.pt"
                unwrapped_model = accelerator.unwrap_model(distilled_model)
                torch.save(unwrapped_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save epoch checkpoint
        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/consistency_model_epoch_{epoch+1}.pt"
            unwrapped_model = accelerator.unwrap_model(distilled_model)
            torch.save(unwrapped_model.state_dict(), checkpoint_path)
            print(f"Saved epoch checkpoint to {checkpoint_path}")
        
        # Reset epoch stats
        total_loss = 0.0
        total_g_loss = 0.0
        total_d_loss = 0.0
    
    # Save final model
    if accelerator.is_main_process:
        final_path = f"{output_dir}/consistency_model_final.pt"
        unwrapped_model = accelerator.unwrap_model(distilled_model)
        torch.save(unwrapped_model.state_dict(), final_path)
        print(f"Saved final consistency model to {final_path}")
    
    # Close wandb
    if use_wandb and accelerator.is_main_process:
        wandb.finish()
    
    # Return the appropriate model
    return accelerator.unwrap_model(distilled_model)


if __name__ == "__main__":
    import argparse
    from wan.configs import t2v_14B
    from accelerate import Accelerator
    
    parser = argparse.ArgumentParser(description="Train consistency distillation for Wan")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for saving checkpoints")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID (for original model)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Whether to use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="wan-consistency-distillation", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--config_file", type=str, default=None, help="Path to OmegaConf config file")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint interval (steps)")
    parser.add_argument("--f_divergence", type=str, default="jensen-shannon", 
                      choices=["reverse-kl", "forward-kl", "jensen-shannon", "squared-hellinger", "softened-rkl"],
                      help="Type of f-divergence to use")
    parser.add_argument("--use_discriminator", type=str2bool, default=True, help="Whether to use discriminator")
    parser.add_argument("--discriminator_lr", type=float, default=1e-5, help="Discriminator learning rate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for GAN loss")
    parser.add_argument("--beta", type=float, default=0.9, help="Weight for f-divergence loss")
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config_file and os.path.exists(args.config_file):
        config = OmegaConf.load(args.config_file)
        # Merge with command line arguments (command line takes precedence)
        args_dict = vars(args)
        for key, value in config.items():
            if key not in args_dict or args_dict[key] is None:
                args_dict[key] = value
        args = argparse.Namespace(**args_dict)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Set device - still needed for the original model and other components
    device = accelerator.device
    
    # Initialize original Wan model - not wrapped by accelerator
    original_model = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        rank=0,
    ).model
    
    # Move original model to accelerator device
    original_model.to(device)
    
    # Create dummy dataset for this example
    # In practice, you'd use your actual training dataset
    from torch.utils.data import TensorDataset, DataLoader
    
    dummy_data = torch.randn(100, 16, 1, 128, 128)  # [N, C, T, H, W]
    dummy_prompts = ["A beautiful landscape"] * 100
    train_dataset = TensorDataset(dummy_data, dummy_prompts)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Train consistency model
    distilled_model = train_consistency_distillation(
        original_model=original_model,
        config=t2v_14B,
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
        f_divergence=args.f_divergence,
        use_discriminator=args.use_discriminator,
        discriminator_lr=args.discriminator_lr,
        alpha=args.alpha,
        beta=args.beta,
    )