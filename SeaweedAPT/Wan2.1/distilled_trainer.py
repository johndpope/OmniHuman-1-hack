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
        learning_rate: Learning rate for optimizer
        cfg_scale: Classifier-free guidance scale
        save_interval: Interval to save checkpoints (in steps)
        use_wandb: Whether to use Weights & Biases for logging
        project_name: WandB project name
        run_name: WandB run name
    
    Returns:
        distilled_model: The trained consistency-distilled model
    """
    print("Initializing consistency distillation training...")
    
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
            }
        )
    
    # Initialize distilled model from scratch with same architecture
    distilled_model = WanModel.from_pretrained(checkpoint_dir)
    
    # Create optimizer
    optimizer = optim.AdamW(distilled_model.parameters(), lr=learning_rate)
    
    # Set up accelerator
    distilled_model, optimizer, train_dataloader = accelerator.prepare(
        distilled_model, optimizer, train_dataloader
    )
    
    # Set both models to appropriate modes
    original_model.eval()
    distilled_model.train()
    
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
    
    # Training loop
    step = 0
    total_loss = 0.0
    
    for epoch in range(num_epochs):
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
            
            # Compute MSE loss
            loss = F.mse_loss(v_student, v_teacher)
            
            # Update with accelerator
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
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
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.6f}")
            
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
    parser.add_argument("--config_file", type=str, default="./config.yaml", help="Path to OmegaConf config file")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint interval (steps)")
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
    train_dataset = TensorDataset(dummy_data, torch.zeros(100))  # Second tensor is a placeholder
    
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
    )