import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Import Wan modules
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

class WanCrossAttentionDiscriminatorBlock(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Create a single learnable token for cross-attention
        self.query_token = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        
        # Cross-attention layers
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Optional query/key normalization
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim, eps=eps)
            self.k_norm = nn.LayerNorm(dim, eps=eps)
        
    def forward(self, x):
        """
        Args:
            x: Visual features from backbone [B, L, C]
        Returns:
            token: Single token output [B, 1, C]
        """
        batch_size = x.shape[0]
        
        # Create batch of query tokens
        query = self.query_token.expand(batch_size, -1, -1)
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Project q, k, v
        q = self.q_proj(query)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Apply query/key normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Reshape for attention
        head_dim = self.dim // self.num_heads
        q = q.view(batch_size, 1, self.num_heads, head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)  # [B, 1, dim]
        
        # Project output
        out = self.o_proj(out)
        
        return out  # [B, 1, dim]


class WanAPTDiscriminator(nn.Module):
    def __init__(self, wan_model):
        super().__init__()
        # Initialize from pre-trained Wan
        self.backbone = copy.deepcopy(wan_model)
        
        # Add cross-attention blocks at layers 16, 26, and 36
        self.cross_attn_16 = WanCrossAttentionDiscriminatorBlock(
            dim=self.backbone.dim,
            num_heads=self.backbone.num_heads,
            qk_norm=self.backbone.qk_norm,
            eps=self.backbone.eps
        )
        self.cross_attn_26 = WanCrossAttentionDiscriminatorBlock(
            dim=self.backbone.dim,
            num_heads=self.backbone.num_heads,
            qk_norm=self.backbone.qk_norm,
            eps=self.backbone.eps
        )
        self.cross_attn_36 = WanCrossAttentionDiscriminatorBlock(
            dim=self.backbone.dim,
            num_heads=self.backbone.num_heads,
            qk_norm=self.backbone.qk_norm,
            eps=self.backbone.eps
        )
        
        # Project concatenated features to a scalar logit
        self.final_proj = nn.Sequential(
            nn.LayerNorm(wan_model.dim * 3),
            nn.Linear(wan_model.dim * 3, 1)
        )
    
    def forward(self, x, t, context, seq_len, return_features=False):
        """
        Discriminator forward pass, extracting features from specific layers
        
        Args:
            x: Latent input [B, C, T, H, W]
            t: Diffusion timestep
            context: Text embedding
            seq_len: Maximum sequence length
            return_features: Whether to return intermediate features
        
        Returns:
            logit: Scalar output for real/fake classification
        """
        # Process input through backbone up to layer 16
        features = []
        layer_outputs = {}
        
        # We need to hook into the forward computation of the backbone
        # to extract intermediate layer outputs
        def extract_features(module, input, output, layer_idx):
            layer_outputs[layer_idx] = output
        
        # Register hooks for layers 16, 26, 36
        hooks = []
        hooks.append(self.backbone.blocks[15].register_forward_hook(
            lambda m, i, o: extract_features(m, i, o, 16)))
        hooks.append(self.backbone.blocks[25].register_forward_hook(
            lambda m, i, o: extract_features(m, i, o, 26)))
        hooks.append(self.backbone.blocks[35].register_forward_hook(
            lambda m, i, o: extract_features(m, i, o, 36)))
        
        # Forward pass through backbone
        with torch.no_grad():
            self.backbone(x, t, context, seq_len)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Apply cross-attention blocks to extracted features
        feat_16 = self.cross_attn_16(layer_outputs[16])
        feat_26 = self.cross_attn_26(layer_outputs[26])
        feat_36 = self.cross_attn_36(layer_outputs[36])
        
        # Concatenate features and project to logit
        concat_features = torch.cat([
            feat_16.squeeze(1), 
            feat_26.squeeze(1), 
            feat_36.squeeze(1)
        ], dim=-1)
        
        logit = self.final_proj(concat_features)
        
        if return_features:
            return logit, [feat_16, feat_26, feat_36]
        else:
            return logit


class WanAPTGenerator(nn.Module):
    def __init__(self, distilled_model, final_timestep=1000):
        super().__init__()
        # Initialize from distilled model
        self.model = distilled_model
        self.final_timestep = final_timestep
    
    def forward(self, z, context, seq_len):
        """
        One-step generator forward pass
        
        Args:
            z: Noise input
            context: Text embedding
            seq_len: Maximum sequence length
        
        Returns:
            x: Generated latents
        """
        # Always use final timestep for one-step generation
        t = torch.ones(z.shape[0], device=z.device) * self.final_timestep
        
        # Predict velocity field
        v = self.model(z, t, context, seq_len)
        
        # Convert to sample
        x = z - v
        
        return x


def approximated_r1_loss(discriminator, real_samples, timestep, context, seq_len, sigma=0.01):
    """
    Compute approximated R1 regularization loss
    
    Args:
        discriminator: Discriminator model
        real_samples: Real latent samples
        timestep: Diffusion timestep
        context: Text embedding
        seq_len: Maximum sequence length
        sigma: Standard deviation for perturbation
    
    Returns:
        loss: Approximated R1 loss
    """
    # Get discriminator prediction on real samples
    real_pred = discriminator(real_samples, timestep, context, seq_len)
    
    # Add small Gaussian perturbation to real samples
    perturbed_samples = real_samples + torch.randn_like(real_samples) * sigma
    
    # Get discriminator prediction on perturbed samples
    perturbed_pred = discriminator(perturbed_samples, timestep, context, seq_len)
    
    # Compute approximated R1 loss
    loss = torch.mean((real_pred - perturbed_pred) ** 2)
    
    return loss


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model)
        self.decay = decay
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)


def train_seaweed_apt(config, 
                      original_model,
                      distilled_model, 
                      train_dataloader_image, 
                      train_dataloader_video,
                      device,
                      checkpoint_dir,
                      distributed=True):
    """
    Train Seaweed APT with Wan as the base model
    
    Args:
        config: Training configuration
        original_model: Original pre-trained Wan model
        distilled_model: Consistency-distilled Wan model
        train_dataloader_image: DataLoader for image training
        train_dataloader_video: DataLoader for video training
        device: Training device
        checkpoint_dir: Directory for saving checkpoints
        distributed: Whether to use distributed training
    """
    # Initialize generator and discriminator
    generator = WanAPTGenerator(distilled_model, final_timestep=config.num_train_timesteps)
    discriminator = WanAPTDiscriminator(original_model)
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Wrap with FSDP if distributed
    if distributed:
        generator = FSDP(generator, device_id=device)
        discriminator = FSDP(discriminator, device_id=device)
    
    # Setup optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), 
                                 lr=config.g_lr_image, 
                                 alpha=0.9)
    d_optimizer = optim.RMSprop(discriminator.parameters(), 
                                 lr=config.d_lr_image, 
                                 alpha=0.9)
    
    # Setup gradient scaler for mixed precision
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    
    # Setup EMA
    ema = EMA(generator, decay=config.ema_decay)
    
    # Phase 1: Image Training
    print("Starting image training phase...")
    for update in range(config.image_updates):
        for batch_idx, (images, text_prompts) in enumerate(train_dataloader_image):
            # Move data to device
            images = images.to(device)
            
            # Process text prompts through T5 encoder
            text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=device,
                checkpoint_path=f"{checkpoint_dir}/{config.t5_checkpoint}",
                tokenizer_path=f"{checkpoint_dir}/{config.t5_tokenizer}"
            )
            context = text_encoder(text_prompts, device)
            
            # Sample timestep with shift function
            t = torch.rand(images.shape[0], device=device) * config.num_train_timesteps
            s = 1.0  # For images
            t_shifted = s * t / (1.0 + (s - 1.0) * t)
            
            # Train discriminator
            with autocast():
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
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()
            
            # Train generator
            with autocast():
                # Generate fake samples
                fake_images = generator(noise, context, config.seq_len)
                
                # Discriminator predictions on new fake samples
                fake_logits = discriminator(fake_images, t_shifted, context, config.seq_len)
                
                # Compute generator loss
                g_loss = -torch.mean(torch.log(torch.sigmoid(fake_logits) + 1e-8))
            
            # Update generator
            g_optimizer.zero_grad()
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
            
            # Update EMA model
            ema.update(generator)
            
            if batch_idx % 10 == 0:
                print(f"[Image] Update {update}, Batch {batch_idx}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Save checkpoint after each update
        if update % 50 == 0 or update == config.image_updates - 1:
            torch.save({
                'generator': generator.state_dict(),
                'ema_generator': ema.model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'update': update,
            }, f"{checkpoint_dir}/seaweed_apt_image_update_{update}.pt")
    
    # Save final image phase checkpoint
    torch.save({
        'generator': generator.state_dict(),
        'ema_generator': ema.model.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, f"{checkpoint_dir}/seaweed_apt_image_final.pt")
    
    # Phase 2: Video Training
    print("Starting video training phase...")
    
    # Load EMA generator from image phase
    generator = WanAPTGenerator(distilled_model, final_timestep=config.num_train_timesteps)
    checkpoint = torch.load(f"{checkpoint_dir}/seaweed_apt_image_final.pt")
    generator.load_state_dict(checkpoint['ema_generator'])
    generator = generator.to(device)
    
    # Re-initialize discriminator from original weights
    discriminator = WanAPTDiscriminator(original_model)
    discriminator = discriminator.to(device)
    
    # Wrap with FSDP if distributed
    if distributed:
        generator = FSDP(generator, device_id=device)
        discriminator = FSDP(discriminator, device_id=device)
    
    # Setup new optimizers with lower learning rate
    g_optimizer = optim.RMSprop(generator.parameters(), 
                                lr=config.g_lr_video, 
                                alpha=0.9)
    d_optimizer = optim.RMSprop(discriminator.parameters(), 
                                lr=config.d_lr_video, 
                                alpha=0.9)
    
    # Setup EMA for video phase
    ema = EMA(generator, decay=config.ema_decay)
    
    for update in range(config.video_updates):
        for batch_idx, (videos, text_prompts) in enumerate(train_dataloader_video):
            # Move data to device
            videos = videos.to(device)
            
            # Process text prompts through T5 encoder
            text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=device,
                checkpoint_path=f"{checkpoint_dir}/{config.t5_checkpoint}",
                tokenizer_path=f"{checkpoint_dir}/{config.t5_tokenizer}"
            )
            context = text_encoder(text_prompts, device)
            
            # Sample timestep with shift function
            t = torch.rand(videos.shape[0], device=device) * config.num_train_timesteps
            s = 12.0  # For videos
            t_shifted = s * t / (1.0 + (s - 1.0) * t)
            
            # Train discriminator
            with autocast():
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
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()
            
            # Train generator
            with autocast():
                # Generate fake samples
                fake_videos = generator(noise, context, config.seq_len)
                
                # Discriminator predictions on new fake samples
                fake_logits = discriminator(fake_videos, t_shifted, context, config.seq_len)
                
                # Compute generator loss
                g_loss = -torch.mean(torch.log(torch.sigmoid(fake_logits) + 1e-8))
            
            # Update generator
            g_optimizer.zero_grad()
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
            
            # Update EMA model
            ema.update(generator)
            
            if batch_idx % 10 == 0:
                print(f"[Video] Update {update}, Batch {batch_idx}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Save checkpoint after each update
        if update % 50 == 0 or update == config.video_updates - 1:
            torch.save({
                'generator': generator.state_dict(),
                'ema_generator': ema.model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'update': update,
            }, f"{checkpoint_dir}/seaweed_apt_video_update_{update}.pt")
    
    # Save final video phase checkpoint
    torch.save({
        'generator': generator.state_dict(),
        'ema_generator': ema.model.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, f"{checkpoint_dir}/seaweed_apt_video_final.pt")
    
    return ema.model  # Return final EMA generator


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
    # Example usage
    import argparse
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from wan.configs import t2v_14B, SIZE_CONFIGS
    from wan.text2video import WanT2V
    
    parser = argparse.ArgumentParser(description="Train Seaweed APT with Wan base model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for saving checkpoints")
    parser.add_argument("--world_size", type=int, default=1, help="Number of distributed processes")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--consistency_path", type=str, default="", help="Path to pretrained consistency model (optional)")
    args = parser.parse_args()
    
    # Initialize distributed environment
    if args.world_size > 1:
        dist.init_process_group(backend="nccl", world_size=args.world_size, rank=args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cuda:{args.device_id}")
    
    # Define config
    config = SeaweedAPTConfig()
    
    # Initialize original Wan model (T2V-14B)
    original_model = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id if args.world_size == 1 else args.local_rank,
        rank=0 if args.world_size == 1 else args.local_rank,
        t5_fsdp=args.world_size > 1,
        dit_fsdp=args.world_size > 1,
        use_usp=args.world_size > 1
    ).model
    
    # Load or train consistency-distilled model
    if args.consistency_path:
        print(f"Loading pre-trained consistency model from {args.consistency_path}")
        distilled_model = WanModel.from_pretrained(args.consistency_path)
    else:
        print("Training consistency model from scratch...")
        # Here you would implement the consistency distillation training
        # This is a significant process that requires its own training loop
        # For now, we'll just use a copy of the original model as a placeholder
        distilled_model = copy.deepcopy(original_model)
    
    # Setup data loaders (simplified for example)
    # In practice, you'd need proper dataset classes for images and videos
    from torch.utils.data import TensorDataset, DataLoader
    
    # Dummy data for example
    dummy_image_data = torch.randn(100, 16, 1, 128, 128)  # [N, C, T, H, W]
    dummy_image_prompts = ["A beautiful landscape"] * 100
    dummy_video_data = torch.randn(100, 16, 48, 90, 160)  # [N, C, T, H, W]
    dummy_video_prompts = ["A beautiful landscape in motion"] * 100
    
    train_dataset_image = TensorDataset(dummy_image_data)
    train_dataset_video = TensorDataset(dummy_video_data)
    
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
        checkpoint_dir=args.checkpoint_dir,
        distributed=args.world_size > 1
    )
    
    # Save final model
    torch.save(final_model.state_dict(), f"{args.output_dir}/seaweed_wan_apt_final.pt")
    print(f"Training complete! Final model saved to {args.output_dir}/seaweed_wan_apt_final.pt")



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    num_epochs=10,
    learning_rate=1e-5,
    cfg_scale=7.5,
    save_interval=10,
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
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        cfg_scale: Classifier-free guidance scale
        save_interval: Interval to save checkpoints (in steps)
    
    Returns:
        distilled_model: The trained consistency-distilled model
    """
    print("Initializing consistency distillation training...")
    
    # Initialize distilled model from scratch with same architecture
    distilled_model = WanModel.from_pretrained(checkpoint_dir)
    distilled_model.to(device)
    
    # Set both models to evaluation mode
    original_model.eval()
    distilled_model.train()
    
    # Create optimizer
    optimizer = optim.AdamW(distilled_model.parameters(), lr=learning_rate)
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    
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
            # Move batch to device
            samples = samples.to(device)
            
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
                with autocast():
                    v_uncond = original_model(
                        [noise], 
                        t=timestep, 
                        context=context_null, 
                        seq_len=config.seq_len
                    )[0]
                
                # Conditional prediction
                with autocast():
                    v_cond = original_model(
                        [noise], 
                        t=timestep, 
                        context=context, 
                        seq_len=config.seq_len
                    )[0]
                
                # Apply classifier-free guidance
                v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Compute student prediction
            with autocast():
                v_student = distilled_model(
                    [noise], 
                    t=timestep, 
                    context=context, 
                    seq_len=config.seq_len
                )[0]
            
            # Compute MSE loss
            loss = F.mse_loss(v_student, v_teacher)
            
            # Update model with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update stats
            total_loss += loss.item()
            step += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if step % save_interval == 0:
                checkpoint_path = f"{output_dir}/consistency_model_step_{step}.pt"
                torch.save(distilled_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save epoch checkpoint
        checkpoint_path = f"{output_dir}/consistency_model_epoch_{epoch+1}.pt"
        torch.save(distilled_model.state_dict(), checkpoint_path)
        print(f"Saved epoch checkpoint to {checkpoint_path}")
        
        # Reset epoch stats
        total_loss = 0.0
    
    # Save final model
    final_path = f"{output_dir}/consistency_model_final.pt"
    torch.save(distilled_model.state_dict(), final_path)
    print(f"Saved final consistency model to {final_path}")
    
    return distilled_model


if __name__ == "__main__":
    import argparse
    from wan.configs import t2v_14B
    
    parser = argparse.ArgumentParser(description="Train consistency distillation for Wan")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for saving checkpoints")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.device_id}")
    
    # Initialize original Wan model
    original_model = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        rank=0,
    ).model
    
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
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cfg_scale=args.cfg_scale,
    )



import argparse
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from wan.configs import t2v_14B, SIZE_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.utils import cache_video

class SeaweedWanAPTGenerator:
    def __init__(
        self,
        model_path,
        checkpoint_dir,
        device_id=0,
        multi_gpu=False,
        num_gpus=1,
    ):
        """
        Initialize the Seaweed WAN APT one-step video generator
        
        Args:
            model_path: Path to the trained one-step generator model
            checkpoint_dir: Directory with Wan checkpoints
            device_id: CUDA device ID
            multi_gpu: Whether to use multiple GPUs for inference
            num_gpus: Number of GPUs to use for parallel inference
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.multi_gpu = multi_gpu
        self.num_gpus = num_gpus
        self.config = t2v_14B
        
        # Load model components
        print(f"Loading components...")
        
        # Text encoder
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),  # Load on CPU first
            checkpoint_path=os.path.join(checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, self.config.t5_tokenizer),
        )
        
        # VAE
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, self.config.vae_checkpoint),
            device=self.device
        )
        
        # One-step generator
        self.model = WanModel.from_pretrained(checkpoint_dir)
        # Load trained weights
        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        # Move models to devices
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def generate(
        self,
        prompt,
        size=(1280, 720),
        frame_num=49,  # 2 seconds at 24fps + 1 frame
        negative_prompt="",
        seed=-1,
        save_path=None,
    ):
        """
        Generate a video using one-step inference
        
        Args:
            prompt: Text prompt for generation
            size: Video dimensions (width, height)
            frame_num: Number of frames to generate
            negative_prompt: Negative prompt for generation
            seed: Random seed (-1 for random)
            save_path: Path to save the generated video
            
        Returns:
            Generated video tensor
        """
        start_time = time.time()
        
        # Set seed
        if seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print(f"Using seed: {seed}")
        
        # Setup random generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        # Process text prompt
        print("Processing text prompt...")
        text_encode_start = time.time()
        
        # Move text encoder to device for inference
        self.text_encoder.model.to(self.device)
        
        context = self.text_encoder([prompt], self.device)
        
        if negative_prompt:
            context_null = self.text_encoder([negative_prompt], self.device)
        else:
            context_null = self.text_encoder([self.config.sample_neg_prompt], self.device)
        
        # Offload text encoder back to CPU
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        
        text_encode_time = time.time() - text_encode_start
        print(f"Text encoding completed in {text_encode_time:.2f}s")
        
        # Calculate latent dimensions
        F = frame_num
        target_shape = (
            self.vae.model.z_dim, 
            (F - 1) // self.config.vae_stride[0] + 1,
            size[1] // self.config.vae_stride[1],
            size[0] // self.config.vae_stride[2]
        )
        
        # Calculate sequence length
        seq_len = (
            (target_shape[2] * target_shape[3]) // 
            (self.config.patch_size[1] * self.config.patch_size[2]) * 
            target_shape[1]
        )
        
        # Generate latent noise
        print(f"Generating latent noise with shape {target_shape}...")
        noise = torch.randn(
            target_shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator
        ).unsqueeze(0)  # Add batch dimension
        
        # One-step inference
        print("Performing one-step inference...")
        dit_start = time.time()
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.config.param_dtype):
            latents = self.model(
                [noise], 
                t=torch.ones(1, device=self.device) * self.config.num_train_timesteps,
                context=context,
                seq_len=seq_len
            )[0]
        
        dit_time = time.time() - dit_start
        print(f"DiT inference completed in {dit_time:.2f}s")
        
        # Decode latents to video
        print("Decoding latents to video...")
        vae_start = time.time()
        
        with torch.no_grad():
            videos = self.vae.decode([latents])
        
        vae_time = time.time() - vae_start
        print(f"VAE decoding completed in {vae_time:.2f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"Total generation time: {total_time:.2f}s")
        
        # Save video if path is provided
        if save_path:
            print(f"Saving video to {save_path}...")
            save_video(videos[0], save_path)
        
        return videos[0]
    
    def __del__(self):
        # Clean up resources
        try:
            del self.model
            del self.text_encoder
            del self.vae
            torch.cuda.empty_cache()
        except:
            pass


def save_video(video_tensor, save_path, fps=24):
    """
    Save video tensor to file
    
    Args:
        video_tensor: Video tensor [C, T, H, W]
        save_path: Path to save the video
        fps: Frames per second
    """
    # Use wan utility function for saving video
    cache_video(
        video_tensor.unsqueeze(0),  # Add batch dimension
        save_file=save_path,
        fps=fps,
        normalize=True,
        value_range=(-1, 1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-step video generation with Seaweed-Wan")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained one-step generator model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with Wan checkpoints")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--frames", type=int, default=49, help="Number of frames (2s at 24fps + 1 frame)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs for inference")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Initialize generator
    generator = SeaweedWanAPTGenerator(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus,
    )
    
    # Generate video
    video = generator.generate(
        prompt=args.prompt,
        size=(args.width, args.height),
        frame_num=args.frames,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        save_path=args.output_path,
    )
    
    print(f"Video generation complete! Saved to {args.output_path}")