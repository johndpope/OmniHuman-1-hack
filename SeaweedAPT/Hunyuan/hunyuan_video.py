import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

class APTCrossAttentionBlock(nn.Module):
    """Cross-attention block used in APT discriminator, adapted for HunyuanVideo."""
    def __init__(self, hidden_size: int, heads_num: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        
        # Learnable token that will attend to visual features
        self.learnable_token = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
        )
        
        # Attention components
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, dtype=dtype, device=device)
        self.proj = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Expand learnable token to batch size
        token = self.learnable_token.expand(B, -1, -1)
        
        # Apply normalization
        x = self.norm(x)
        token = self.norm(token)
        
        # Get Q from token, K/V from input features
        q = self.qkv(token)[:, :, :self.hidden_size]
        k = self.qkv(x)[:, :, self.hidden_size:2*self.hidden_size]
        v = self.qkv(x)[:, :, 2*self.hidden_size:]
        
        # Reshape for attention
        q = q.view(B, -1, self.heads_num, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.heads_num, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.heads_num, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.hidden_size)
        x = self.proj(x)
        
        return x

class HunyuanAPTDiscriminator(nn.Module):
    """Discriminator adapting HunyuanVideo architecture for Seaweed APT."""
    def __init__(
        self,
        hyvideo_model,
        cross_attn_layers_double=[10, 18],  # Proportionally placed in double blocks (total 20)
        cross_attn_layers_single=[20, 40],  # Proportionally placed in single blocks (total 40)
        max_timesteps=1000,
        device=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.hyvideo = hyvideo_model
        self.hidden_size = hyvideo_model.hidden_size
        self.heads_num = hyvideo_model.heads_num
        self.max_timesteps = max_timesteps
        
        # Cross-attention blocks for both streams
        self.cross_attn_layers_double = cross_attn_layers_double
        self.cross_attn_layers_single = cross_attn_layers_single
        
        # Create cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            APTCrossAttentionBlock(self.hidden_size, self.heads_num, device, dtype)
            for _ in cross_attn_layers_double + cross_attn_layers_single
        ])
        
        # Final projection to scalar logit
        self.final_norm = nn.LayerNorm(
            self.hidden_size * len(self.cross_attn_blocks), 
            dtype=dtype, 
            device=device
        )
        self.final_proj = nn.Linear(
            self.hidden_size * len(self.cross_attn_blocks), 
            1, 
            dtype=dtype, 
            device=device
        )
        
        # Register hooks to extract intermediate features
        self._register_feature_hooks()

    def _register_feature_hooks(self):
        """Register hooks to collect features from intermediate layers."""
        self.features = []
        self.hooks = []
        
        def _get_hook(layer_idx):
            def _hook(module, input, output):
                if layer_idx in self.cross_attn_layers_double:
                    # For double blocks, output is (img, txt)
                    self.features.append(output[0])  # Take img features
                elif layer_idx in self.cross_attn_layers_single:
                    # For single blocks, output is tensor
                    self.features.append(output)
            return _hook
        
        # Register hooks for double stream blocks
        for i, block in enumerate(self.hyvideo.double_blocks):
            if i in self.cross_attn_layers_double:
                hook = block.register_forward_hook(_get_hook(i))
                self.hooks.append(hook)
        
        # Register hooks for single stream blocks
        for i, block in enumerate(self.hyvideo.single_blocks, start=len(self.hyvideo.double_blocks)):
            if i in self.cross_attn_layers_single:
                hook = block.register_forward_hook(_get_hook(i))
                self.hooks.append(hook)

    def _shift_timestep(self, t: torch.Tensor, shift: float) -> torch.Tensor:
        """Timestep shifting function from Seaweed (Equation 7)."""
        return shift * t / (1 + (shift - 1) * t)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, 
                text_states: Optional[torch.Tensor] = None, 
                text_states_2: Optional[torch.Tensor] = None, 
                is_video: bool = False) -> torch.Tensor:
        # Clear features list
        self.features = []
        
        # Sample timestep uniformly from [0, T] and shift if not provided
        if t is None:
            t = torch.rand(x.shape[0], device=x.device) * self.max_timesteps
            shift = 12.0 if is_video else 1.0
            t = self._shift_timestep(t, shift)
        
        # Forward pass through HunyuanVideo model to extract features via hooks
        _ = self.hyvideo(
            x, 
            t,
            text_states=text_states,
            text_mask=None,  # Assuming masks are handled internally
            text_states_2=text_states_2,
            return_dict=True
        )
        
        # Process features through cross-attention blocks
        cross_attn_outputs = [
            block(feat) for feat, block in zip(self.features, self.cross_attn_blocks)
        ]
        
        # Combine outputs and project to scalar
        combined = torch.cat(cross_attn_outputs, dim=-1)
        combined = self.final_norm(combined)
        return self.final_proj(combined)
        
    def __del__(self):
        # Remove hooks when object is deleted
        for hook in self.hooks:
            hook.remove()

class HunyuanAPTGenerator(nn.Module):
    """One-step generator using HunyuanVideo."""
    def __init__(self, hyvideo_model, max_timesteps: int = 1000):
        super().__init__()
        self.hyvideo = hyvideo_model
        self.max_timesteps = max_timesteps
        
    def forward(self, z: torch.Tensor, text_states: torch.Tensor, 
                text_mask: Optional[torch.Tensor] = None,
                text_states_2: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Use final timestep for one-step generation
        t = torch.full((z.shape[0],), self.max_timesteps, device=z.device)
        
        # Predict velocity through HunyuanVideo
        output = self.hyvideo(
            z, 
            t, 
            text_states=text_states,
            text_mask=text_mask, 
            text_states_2=text_states_2,
            return_dict=True
        )
        
        # Get velocity (output["x"]) and denoise z
        return z - output["x"]

class ApproximatedR1Regularization:
    """Approximated R1 regularization for HunyuanVideo."""
    def __init__(self, sigma: float = 0.01, lambda_r1: float = 100.0):
        self.sigma = sigma
        self.lambda_r1 = lambda_r1
        
    def __call__(
        self, 
        discriminator: HunyuanAPTDiscriminator, 
        x: torch.Tensor, 
        text_states: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        text_states_2: Optional[torch.Tensor] = None,
        is_video: bool = False
    ) -> torch.Tensor:
        # Add small Gaussian noise to real samples
        x_perturbed = x + torch.randn_like(x) * self.sigma
        
        # Get discriminator outputs for real and perturbed samples
        t = torch.rand(x.shape[0], device=x.device) * discriminator.max_timesteps
        shift = 12.0 if is_video else 1.0
        t = discriminator._shift_timestep(t, shift)
        
        d_real = discriminator(x, t, text_states, text_states_2, is_video)
        d_perturbed = discriminator(x_perturbed, t, text_states, text_states_2, is_video)
        
        # Compute approximated R1 loss
        return self.lambda_r1 * torch.mean((d_real - d_perturbed) ** 2)

class EMAModel:
    """Exponential Moving Average model tracking."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone().detach() 
                      for name, param in model.named_parameters() 
                      if param.requires_grad}
        self.backup = {}
        
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].clone()
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()

class HunyuanAPTTrainer:
    """APT Trainer for HunyuanVideo."""
    def __init__(
        self,
        generator: HunyuanAPTGenerator,
        discriminator: HunyuanAPTDiscriminator,
        r1_reg: ApproximatedR1Regularization,
        g_learning_rate: float = 5e-6,
        d_learning_rate: float = 5e-6,
        ema_decay: float = 0.995,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        is_distributed: bool = False,
        world_size: int = 1,
        local_rank: int = 0
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.generator = generator.to(self.device, dtype=self.dtype)
        self.discriminator = discriminator.to(self.device, dtype=self.dtype)
        self.r1_reg = r1_reg
        
        # RMSProp optimizers as specified in Seaweed paper
        self.g_optimizer = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=g_learning_rate,
            alpha=0.9,  # Equivalent to Adam beta2=0.9
            eps=1e-8
        )
        self.d_optimizer = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=d_learning_rate,
            alpha=0.9,
            eps=1e-8
        )
        
        # EMA for generator
        self.ema = EMAModel(self.generator, decay=ema_decay)
        
        # Distributed training setup
        self.is_distributed = is_distributed
        self.world_size = world_size
        self.local_rank = local_rank
        
        if self.is_distributed:
            # Using PyTorch DDP for distributed training
            from torch.nn.parallel import DistributedDataParallel as DDP
            import torch.distributed as dist
            
            self.generator = DDP(self.generator, device_ids=[self.local_rank])
            self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank])
    
    def set_learning_rate(self, g_lr: float, d_lr: float):
        """Update learning rates for generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
    
    def set_r1_sigma(self, sigma: float):
        """Update R1 regularization sigma."""
        self.r1_reg.sigma = sigma
    
    def train_step(
        self,
        real_samples: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        text_states_2: Optional[torch.Tensor] = None,
        is_video: bool = False,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, float]:
        """Execute one training step of APT."""
        real_samples = real_samples.to(self.device, dtype=self.dtype)
        text_states = text_states.to(self.device, dtype=self.dtype)
        if text_mask is not None:
            text_mask = text_mask.to(self.device)
        if text_states_2 is not None:
            text_states_2 = text_states_2.to(self.device, dtype=self.dtype)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(real_samples, device=self.device, dtype=self.dtype)
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            # Generate fake samples
            fake_samples = self.generator(
                noise, 
                text_states, 
                text_mask,
                text_states_2
            )
            
            # Sample timestep for discriminator
            t = torch.rand(real_samples.shape[0], device=self.device) * self.discriminator.max_timesteps
            shift = 12.0 if is_video else 1.0
            t = self.discriminator._shift_timestep(t, shift)
            
            # Compute discriminator losses
            d_real = self.discriminator(
                real_samples, 
                t, 
                text_states, 
                text_states_2, 
                is_video
            )
            d_fake = self.discriminator(
                fake_samples.detach(), 
                t, 
                text_states, 
                text_states_2, 
                is_video
            )
            
            # Non-saturating GAN loss for discriminator
            d_loss = -(torch.mean(F.logsigmoid(d_real)) + torch.mean(F.logsigmoid(-d_fake)))
            
            # Apply approximated R1 regularization
            r1_loss = self.r1_reg(
                self.discriminator, 
                real_samples, 
                text_states, 
                text_mask,
                text_states_2, 
                is_video
            )
            
            d_total_loss = d_loss + r1_loss
        
        d_total_loss.backward()
        self.d_optimizer.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            fake_samples = self.generator(
                noise, 
                text_states, 
                text_mask,
                text_states_2
            )
            
            d_fake = self.discriminator(
                fake_samples, 
                t, 
                text_states, 
                text_states_2, 
                is_video
            )
            
            # Non-saturating GAN loss for generator
            g_loss = -torch.mean(F.logsigmoid(d_fake))
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # Update EMA
        self.ema.update()
        
        # Sync losses across GPUs for distributed training
        if self.is_distributed:
            import torch.distributed as dist
            
            losses = [g_loss.clone(), d_loss.clone(), r1_loss.clone()]
            for loss in losses:
                dist.all_reduce(loss)
                loss /= self.world_size
            g_loss, d_loss, r1_loss = [loss.item() for loss in losses]
        else:
            g_loss, d_loss, r1_loss = g_loss.item(), d_loss.item(), r1_loss.item()
        
        return g_loss, d_loss, r1_loss
    
    @torch.no_grad()
    def sample(
        self,
        text_states: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        text_states_2: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_ema: bool = True,
        is_video: bool = False
    ) -> torch.Tensor:
        """Generate samples using the generator."""
        text_states = text_states.to(self.device, dtype=self.dtype)
        if text_mask is not None:
            text_mask = text_mask.to(self.device)
        if text_states_2 is not None:
            text_states_2 = text_states_2.to(self.device, dtype=self.dtype)
        
        # Use EMA weights for sampling
        if use_ema:
            self.ema.apply_shadow()
        
        # Generate noise if not provided
        if noise is None:
            b = text_states.shape[0]
            # For HunyuanVideo:
            # - Image latents: (B, 1, 16, H/8, W/8)
            # - Video latents: (B, T/4+1, 16, H/8, W/8)
            if is_video:
                # For 2s@24fps, 1280×720 videos, latent shape is:
                t = 48//4 + 1  # T/4+1, where T=48 for 2s@24fps
                h, w = 720//8, 1280//8  # H/8, W/8
            else:
                # For 1024×1024 images, latent shape is:
                t = 1
                h, w = 1024//8, 1024//8
            
            noise = torch.randn(
                (b, t, 16, h, w),
                device=self.device,
                dtype=self.dtype
            )
        
        # Generate samples
        with torch.cuda.amp.autocast(dtype=self.dtype):
            samples = self.generator(
                noise,
                text_states,
                text_mask,
                text_states_2
            )
        
        # Restore original weights
        if use_ema:
            self.ema.restore()
        
        return samples

class HunyuanAPTConfig:
    """Configuration for HunyuanVideo APT training."""
    def __init__(
        self,
        train_images: bool = True,
        train_videos: bool = True,
        image_size: int = 1024,
        video_width: int = 1280,
        video_height: int = 720,
        video_frames: int = 48,  # 2 seconds @ 24fps
        image_batch_size: int = 9062,  # As per Seaweed paper
        video_batch_size: int = 2048,  # As per Seaweed paper
        image_learning_rate: float = 5e-6,
        video_learning_rate: float = 3e-6,
        ema_decay: float = 0.995,
        image_r1_sigma: float = 0.01,
        video_r1_sigma: float = 0.1,
        r1_lambda: float = 100.0,
        image_updates: int = 350,
        video_updates: int = 300,
        max_timesteps: int = 1000, 
        world_size: int = 1,
        local_rank: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        log_interval: int = 10,
        save_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
    ):
        # Training phases
        self.train_images = train_images
        self.train_videos = train_videos
        
        # Data dimensions
        self.image_size = image_size
        self.video_width = video_width
        self.video_height = video_height
        self.video_frames = video_frames
        
        # Training hyperparameters
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size
        self.image_learning_rate = image_learning_rate
        self.video_learning_rate = video_learning_rate
        self.ema_decay = ema_decay
        self.image_r1_sigma = image_r1_sigma
        self.video_r1_sigma = video_r1_sigma
        self.r1_lambda = r1_lambda
        self.image_updates = image_updates
        self.video_updates = video_updates
        self.max_timesteps = max_timesteps
        
        # Distributed training
        self.world_size = world_size
        self.local_rank = local_rank
        
        # Technical settings
        self.device = device
        self.dtype = dtype
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if self.local_rank == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

def setup_apt_for_hyvideo(
    hyvideo_model,
    distilled_model=None,  # Optional pre-distilled model
    config=None,  # Optional config
):
    """Set up APT training for HunyuanVideo."""
    if config is None:
        config = HunyuanAPTConfig()
    
    # If no distilled model is provided, use the same as original
    if distilled_model is None:
        import copy
        distilled_model = copy.deepcopy(hyvideo_model)
    
    # Create generator and discriminator
    generator = HunyuanAPTGenerator(
        distilled_model,
        max_timesteps=config.max_timesteps
    )
    
    discriminator = HunyuanAPTDiscriminator(
        hyvideo_model,
        cross_attn_layers_double=[5, 10, 15],  # Positioned throughout the 20 double blocks
        cross_attn_layers_single=[25, 35, 45],  # Positioned throughout the 40 single blocks
        max_timesteps=config.max_timesteps,
        device=config.device,
        dtype=config.dtype
    )
    
    # Create R1 regularization
    r1_reg = ApproximatedR1Regularization(
        sigma=config.image_r1_sigma,
        lambda_r1=config.r1_lambda
    )
    
    # Create trainer
    trainer = HunyuanAPTTrainer(
        generator=generator,
        discriminator=discriminator,
        r1_reg=r1_reg,
        g_learning_rate=config.image_learning_rate,
        d_learning_rate=config.image_learning_rate,
        ema_decay=config.ema_decay,
        device=config.device,
        dtype=config.dtype,
        is_distributed=config.world_size > 1,
        world_size=config.world_size,
        local_rank=config.local_rank
    )
    
    return trainer

def distill_hyvideo(hyvideo_model, train_dataloader, validation_dataloader=None, 
                   learning_rate=1e-5, num_steps=1000, device="cuda", dtype=torch.bfloat16):
    """Perform deterministic distillation on HunyuanVideo model.
    
    Args:
        hyvideo_model: Original HunyuanVideo model
        train_dataloader: DataLoader for training data
        validation_dataloader: Optional DataLoader for validation
        learning_rate: Learning rate for optimizer
        num_steps: Number of training steps
        device: Device to use for training
        dtype: Data type for training
        
    Returns:
        Distilled HunyuanVideo model
    """
    import copy
    from torch.optim import Adam
    
    # Create a copy of the model for distillation
    distilled_model = copy.deepcopy(hyvideo_model)
    distilled_model.to(device, dtype)
    hyvideo_model.to(device, dtype)
    hyvideo_model.eval()  # Set original model to eval mode
    
    # Freeze teacher model
    for param in hyvideo_model.parameters():
        param.requires_grad = False
    
    # Set up optimizer
    optimizer = Adam(distilled_model.parameters(), lr=learning_rate)
    
    # Training loop
    distilled_model.train()
    
    # Progress tracking
    progress_bar = tqdm(range(num_steps), desc="Distillation")
    running_loss = 0.0
    
    for step in range(num_steps):
        # Get batch
        for batch in train_dataloader:
            # Unpack batch - adjust based on your dataloader format
            noise, text_states, text_mask, text_states_2 = batch
            
            # Move to device
            noise = noise.to(device, dtype=dtype)
            text_states = text_states.to(device, dtype)
            if text_mask is not None:
                text_mask = text_mask.to(device)
            if text_states_2 is not None:
                text_states_2 = text_states_2.to(device, dtype)
            
            # Get teacher prediction
            with torch.no_grad():
                t = torch.full((noise.shape[0],), hyvideo_model.config.num_train_timesteps, device=device)
                teacher_output = hyvideo_model(
                    noise, 
                    t, 
                    text_states=text_states,
                    text_mask=text_mask, 
                    text_states_2=text_states_2,
                    return_dict=True
                )
                teacher_v = teacher_output["x"]
            
            # Get student prediction
            student_output = distilled_model(
                noise, 
                t, 
                text_states=text_states,
                text_mask=text_mask, 
                text_states_2=text_states_2,
                return_dict=True
            )
            student_v = student_output["x"]
            
            # Compute MSE loss
            loss = F.mse_loss(student_v, teacher_v)
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            break  # Process only one batch per step
        
        # Update progress bar
        if step % 10 == 0:
            progress_bar.set_postfix({"loss": running_loss / 10})
            running_loss = 0.0
            
        progress_bar.update(1)
        
        # Validation
        if validation_dataloader is not None and step % 100 == 0:
            distilled_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in validation_dataloader:
                    # Unpack batch
                    val_noise, val_text_states, val_text_mask, val_text_states_2 = val_batch
                    
                    # Move to device
                    val_noise = val_noise.to(device, dtype=dtype)
                    val_text_states = val_text_states.to(device, dtype)
                    if val_text_mask is not None:
                        val_text_mask = val_text_mask.to(device)
                    if val_text_states_2 is not None:
                        val_text_states_2 = val_text_states_2.to(device, dtype)
                    
                    # Get teacher prediction
                    val_t = torch.full((val_noise.shape[0],), hyvideo_model.config.num_train_timesteps, device=device)
                    val_teacher_output = hyvideo_model(
                        val_noise, 
                        val_t, 
                        text_states=val_text_states,
                        text_mask=val_text_mask, 
                        text_states_2=val_text_states_2,
                        return_dict=True
                    )
                    val_teacher_v = val_teacher_output["x"]
                    
                    # Get student prediction
                    val_student_output = distilled_model(
                        val_noise, 
                        val_t, 
                        text_states=val_text_states,
                        text_mask=val_text_mask, 
                        text_states_2=val_text_states_2,
                        return_dict=True
                    )
                    val_student_v = val_student_output["x"]
                    
                    # Compute MSE loss
                    val_loss += F.mse_loss(val_student_v, val_teacher_v).item()
                    break  # Process only one validation batch

            progress_bar.set_postfix({"loss": running_loss / 10, "val_loss": val_loss})
            distilled_model.train()
   
    # Final evaluation
    if validation_dataloader is not None:
       distilled_model.eval()
       final_val_loss = 0.0
       num_batches = 0
       with torch.no_grad():
           for val_batch in validation_dataloader:
               # Unpack batch
               val_noise, val_text_states, val_text_mask, val_text_states_2 = val_batch
               
               # Move to device
               val_noise = val_noise.to(device, dtype=dtype)
               val_text_states = val_text_states.to(device, dtype)
               if val_text_mask is not None:
                   val_text_mask = val_text_mask.to(device)
               if val_text_states_2 is not None:
                   val_text_states_2 = val_text_states_2.to(device, dtype)
               
               # Get teacher prediction
               val_t = torch.full((val_noise.shape[0],), hyvideo_model.config.num_train_timesteps, device=device)
               val_teacher_output = hyvideo_model(
                   val_noise, 
                   val_t, 
                   text_states=val_text_states,
                   text_mask=val_text_mask, 
                   text_states_2=val_text_states_2,
                   return_dict=True
               )
               val_teacher_v = val_teacher_output["x"]
               
               # Get student prediction
               val_student_output = distilled_model(
                   val_noise, 
                   val_t, 
                   text_states=val_text_states,
                   text_mask=val_text_mask, 
                   text_states_2=val_text_states_2,
                   return_dict=True
               )
               val_student_v = val_student_output["x"]
               
               # Compute MSE loss
               final_val_loss += F.mse_loss(val_student_v, val_teacher_v).item()
               num_batches += 1
               if num_batches >= 10:  # Limit validation to 10 batches
                   break
       
       print(f"Final validation loss: {final_val_loss / num_batches:.6f}")
   
    return distilled_model

class HunyuanAPTLogger:
   """Logger for HunyuanVideo APT training."""
   def __init__(self, config: HunyuanAPTConfig):
       self.config = config
       self.reset_metrics()
       if config.local_rank == 0:  # Only log on the main process
           logging.basicConfig(
               level=logging.INFO,
               format='%(asctime)s [%(levelname)s] %(message)s',
               handlers=[
                   logging.StreamHandler(),
                   logging.FileHandler(os.path.join(config.checkpoint_dir, "training.log"))
               ]
           )
   
   def reset_metrics(self):
       """Reset tracking metrics."""
       self.metrics = {
           "g_loss": [],
           "d_loss": [],
           "r1_loss": []
       }
   
   def log_metrics(self, g_loss: float, d_loss: float, r1_loss: float, step: int, phase: str):
       """Log training metrics."""
       self.metrics["g_loss"].append(g_loss)
       self.metrics["d_loss"].append(d_loss)
       self.metrics["r1_loss"].append(r1_loss)
       
       if self.config.local_rank == 0 and step % self.config.log_interval == 0:
           # Calculate average over last log_interval steps
           log_interval = min(self.config.log_interval, len(self.metrics["g_loss"]))
           avg_g_loss = np.mean(self.metrics["g_loss"][-log_interval:])
           avg_d_loss = np.mean(self.metrics["d_loss"][-log_interval:])
           avg_r1_loss = np.mean(self.metrics["r1_loss"][-log_interval:])
           
           logging.info(
               f"{phase} Step {step}: "
               f"G_loss = {avg_g_loss:.4f}, "
               f"D_loss = {avg_d_loss:.4f}, "
               f"R1_loss = {avg_r1_loss:.4f}"
           )

class HunyuanAPTCheckpointer:
   """Checkpoint manager for HunyuanVideo APT training."""
   def __init__(self, config: HunyuanAPTConfig):
       self.config = config
   
   def save_checkpoint(self, trainer: HunyuanAPTTrainer, step: int, phase: str, is_final: bool = False):
       """Save training checkpoint."""
       if self.config.local_rank == 0:  # Only save on the main process
           # Create checkpoint dictionary
           checkpoint = {
               'generator_state_dict': trainer.generator.module.state_dict() 
                   if trainer.is_distributed else trainer.generator.state_dict(),
               'discriminator_state_dict': trainer.discriminator.module.state_dict() 
                   if trainer.is_distributed else trainer.discriminator.state_dict(),
               'g_optimizer_state_dict': trainer.g_optimizer.state_dict(),
               'd_optimizer_state_dict': trainer.d_optimizer.state_dict(),
               'ema_shadow': trainer.ema.shadow,
               'step': step,
               'phase': phase,
           }
           
           # Determine checkpoint filename
           suffix = "final" if is_final else f"step_{step}"
           checkpoint_path = os.path.join(
               self.config.checkpoint_dir, 
               f"apt_{phase}_{suffix}.pt"
           )
           
           # Save checkpoint
           torch.save(checkpoint, checkpoint_path)
           logging.info(f"Saved checkpoint to {checkpoint_path}")
   
   def load_checkpoint(self, trainer: HunyuanAPTTrainer, path: str) -> Tuple[int, str]:
       """Load training checkpoint."""
       checkpoint = torch.load(path, map_location=self.config.device)
       
       # Load model weights
       if trainer.is_distributed:
           trainer.generator.module.load_state_dict(checkpoint['generator_state_dict'])
           trainer.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
       else:
           trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
           trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
       
       # Load optimizer states
       trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
       trainer.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
       
       # Load EMA weights
       trainer.ema.shadow = checkpoint['ema_shadow']
       
       logging.info(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
       
       return checkpoint['step'], checkpoint['phase']

class HunyuanAPTTrainingLoop:
   """Training loop for HunyuanVideo APT."""
   def __init__(
       self,
       config: HunyuanAPTConfig,
       trainer: HunyuanAPTTrainer,
       image_dataloader=None,
       video_dataloader=None
   ):
       self.config = config
       self.trainer = trainer
       self.image_dataloader = image_dataloader
       self.video_dataloader = video_dataloader
       self.logger = HunyuanAPTLogger(config)
       self.checkpointer = HunyuanAPTCheckpointer(config)
       
       # Initialize distributed training if needed
       if config.world_size > 1 and not torch.distributed.is_initialized():
           import torch.distributed as dist
           dist.init_process_group("nccl")
           torch.cuda.set_device(config.local_rank)
   
   def train_images(self):
       """Train on images first (Section 3.5 of Seaweed paper)."""
       if not self.config.train_images or self.image_dataloader is None:
           return
       
       self.logger.reset_metrics()
       self.trainer.set_learning_rate(self.config.image_learning_rate, self.config.image_learning_rate)
       self.trainer.set_r1_sigma(self.config.image_r1_sigma)
       
       # Progress tracking
       iterator = iter(self.image_dataloader)
       progress_bar = tqdm(
           range(self.config.image_updates),
           desc="Image Training",
           disable=self.config.local_rank != 0
       )
       
       for step in progress_bar:
           try:
               # Get next batch
               batch = next(iterator)
           except StopIteration:
               # Restart iterator if exhausted
               iterator = iter(self.image_dataloader)
               batch = next(iterator)
           
           # Unpack batch - adjust based on your dataloader format
           # Expected format: real_latents, text_states, text_mask, text_states_2
           real_latents, text_states, text_mask, text_states_2 = batch
           
           # Add singleton time dimension for images if necessary
           if len(real_latents.shape) == 4:  # [B, C, H, W]
               real_latents = real_latents.unsqueeze(1)  # [B, 1, C, H, W]
           
           # Train step
           g_loss, d_loss, r1_loss = self.trainer.train_step(
               real_latents,
               text_states,
               text_mask,
               text_states_2,
               is_video=False
           )
           
           # Log metrics
           self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "Image")
           
           # Save checkpoint
           if (step + 1) % self.config.save_interval == 0:
               self.checkpointer.save_checkpoint(self.trainer, step, "image")
       
       # Save final checkpoint
       self.checkpointer.save_checkpoint(
           self.trainer,
           self.config.image_updates,
           "image",
           is_final=True
       )
   
   def train_videos(self):
       """Train on videos after images (Section 3.5 of Seaweed paper)."""
       if not self.config.train_videos or self.video_dataloader is None:
           return
       
       self.logger.reset_metrics()
       self.trainer.set_learning_rate(self.config.video_learning_rate, self.config.video_learning_rate)
       self.trainer.set_r1_sigma(self.config.video_r1_sigma)
       
       # Progress tracking
       iterator = iter(self.video_dataloader)
       progress_bar = tqdm(
           range(self.config.video_updates),
           desc="Video Training",
           disable=self.config.local_rank != 0
       )
       
       for step in progress_bar:
           try:
               # Get next batch
               batch = next(iterator)
           except StopIteration:
               # Restart iterator if exhausted
               iterator = iter(self.video_dataloader)
               batch = next(iterator)
           
           # Unpack batch - adjust based on your dataloader format
           # Expected format: real_latents, text_states, text_mask, text_states_2
           real_latents, text_states, text_mask, text_states_2 = batch
           
           # Train step
           g_loss, d_loss, r1_loss = self.trainer.train_step(
               real_latents,
               text_states,
               text_mask,
               text_states_2,
               is_video=True
           )
           
           # Log metrics
           self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "Video")
           
           # Save checkpoint
           if (step + 1) % self.config.save_interval == 0:
               self.checkpointer.save_checkpoint(self.trainer, step, "video")
       
       # Save final checkpoint
       self.checkpointer.save_checkpoint(
           self.trainer,
           self.config.video_updates,
           "video",
           is_final=True
       )
   
   def train(self):
       """Run the complete training pipeline."""
       # Train on images first
       self.train_images()
       
       # Train on videos after images
       # Note: For video training, ideally we would initialize the generator
       # from the EMA checkpoint of the image training phase as mentioned in the paper
       self.train_videos()

def full_apt_pipeline_for_hyvideo(
   hyvideo_model,
   image_dataloader,
   video_dataloader,
   config=None,
   resume_from=None
):
   """Run the complete APT pipeline for HunyuanVideo.
   
   Args:
       hyvideo_model: HunyuanVideo model
       image_dataloader: DataLoader for image training
       video_dataloader: DataLoader for video training
       config: Optional configuration
       resume_from: Optional path to resume training from checkpoint
       
   Returns:
       Trained generator model
   """
   if config is None:
       config = HunyuanAPTConfig()
   
   # 1. Perform deterministic distillation on HunyuanVideo
   logging.info("Starting deterministic distillation...")
   distilled_model = distill_hyvideo(
       hyvideo_model,
       image_dataloader,  # Use image data for distillation
       learning_rate=1e-5,
       num_steps=1000,
       device=config.device,
       dtype=config.dtype
   )
   
   # 2. Set up APT trainer
   logging.info("Setting up adversarial post-training...")
   trainer = setup_apt_for_hyvideo(
       hyvideo_model,
       distilled_model,
       config
   )
   
   # 3. Create training loop
   training_loop = HunyuanAPTTrainingLoop(
       config=config,
       trainer=trainer,
       image_dataloader=image_dataloader,
       video_dataloader=video_dataloader
   )
   
   # 4. Resume from checkpoint if provided
   if resume_from is not None:
       step, phase = training_loop.checkpointer.load_checkpoint(trainer, resume_from)
       logging.info(f"Resuming training from {phase} phase, step {step}")
   
   # 5. Run training
   logging.info("Starting APT training...")
   training_loop.train()
   
   # 6. Return trained generator
   return trainer.generator

# Example usage
if __name__ == "__main__":
   from hyvideo.config import parse_args
   from hyvideo.modules.models import HYVideoDiffusionTransformer
   
   # Parse arguments
   args = parse_args()
   
   # Create HunyuanVideo model
   hyvideo_model = HYVideoDiffusionTransformer(
       args=args,
       in_channels=16,
       out_channels=16,
   ).to("cuda", dtype=torch.bfloat16)
   
   # Load model weights
   # ... (load weights from checkpoint)
   
   # Create dataloaders
   # ... (create dataloaders for image and video data)
   
   # Run APT pipeline
   config = HunyuanAPTConfig(
       train_images=True,
       train_videos=True,
       image_learning_rate=5e-6,
       video_learning_rate=3e-6,
       image_updates=350,
       video_updates=300,
   )
   
   # Use smaller batch sizes for testing
   image_dataloader = torch.utils.data.DataLoader(
       # Your image dataset here
       batch_size=4,
       shuffle=True,
   )
   
   video_dataloader = torch.utils.data.DataLoader(
       # Your video dataset here
       batch_size=2,
       shuffle=True,
   )
   
   # Run APT pipeline
   generator = full_apt_pipeline_for_hyvideo(
       hyvideo_model,
       image_dataloader,
       video_dataloader,
       config
   )
   
   # Save final model
   torch.save(generator.state_dict(), "hyvideo_apt_generator.pt")