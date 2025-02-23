import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from mmditx import MMDiTX, DismantledBlock, PatchEmbed, TimestepEmbedder, VectorEmbedder

class APTCrossAttentionBlock(nn.Module):
    """Cross-attention block used in APT discriminator."""
    def __init__(self, hidden_size: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Learnable token that will attend to visual features 
        self.learnable_token = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
        )
        
        # Attention components
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype, device=device)
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
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.hidden_size)
        x = self.proj(x)
        
        return x

class APTDiscriminator(nn.Module):
    """Discriminator using MMDiT backbone with modifications (Section 3.3)."""
    def __init__(
        self,
        mmdit_model: MMDiTX,
        cross_attn_layers: List[int] = [16, 26, 36],  # Paper-specified layers
        max_timesteps: int = 1000,  # Adjust based on pre-trained model
        device=None,
        dtype=torch.bfloat16  # Use BF16 as per Section 3.5
    ):
        super().__init__()
        self.mmdit = mmdit_model
        self.hidden_size = mmdit_model.diffusion_model.hidden_size
        self.max_timesteps = max_timesteps
        
        self.cross_attn_blocks = nn.ModuleList([
            APTCrossAttentionBlock(self.hidden_size, mmdit_model.num_heads, device, dtype)
            for _ in cross_attn_layers
        ])
        self.cross_attn_layers = cross_attn_layers
        
        self.final_norm = nn.LayerNorm(self.hidden_size * len(cross_attn_layers), dtype=dtype, device=device)
        self.final_proj = nn.Linear(self.hidden_size * len(cross_attn_layers), 1, dtype=dtype, device=device)

    def _shift_timestep(self, t: torch.Tensor, shift: float) -> torch.Tensor:
        """Timestep shifting (Section 3.3, Equation 7)."""
        return shift * t / (1 + (shift - 1) * t)

    def _get_features(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> List[torch.Tensor]:
        # Adjusted for provided MMDiTX structure
        features = []
        x = self.mmdit.x_embedder(x.squeeze(1) if x.dim() == 5 else x)  # Handle video/image input
        c_mod = self.mmdit.t_embedder(t, dtype=x.dtype)
        if c is not None and hasattr(self.mmdit, 'y_embedder'):
            y = self.mmdit.y_embedder(c)
            c_mod = c_mod + y
        context = self.mmdit.context_embedder(c) if c is not None else None
        for i, block in enumerate(self.mmdit.joint_blocks):
            context, x = block(context, x, c=c_mod)
            if i in self.cross_attn_layers:
                features.append(x)
        return features

    def forward(self, x: torch.Tensor, c: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        # Sample timestep uniformly from [0, T] and shift (Section 3.3)
        t = torch.rand(x.shape[0], device=x.device) * self.max_timesteps
        shift = 12.0 if is_video else 1.0
        t = self._shift_timestep(t, shift)
        
        features = self._get_features(x, c, t)
        cross_attn_outputs = [block(feat) for feat, block in zip(features, self.cross_attn_blocks)]
        
        combined = torch.cat(cross_attn_outputs, dim=-1)
        combined = self.final_norm(combined)
        return self.final_proj(combined)
    

class APTGenerator(nn.Module):
    """One-step generator from distilled MMDiT (Section 3.2)."""
    def __init__(self, mmdit_model: MMDiTX, max_timesteps: int = 1000):
        super().__init__()
        self.mmdit = mmdit_model
        self.max_timesteps = max_timesteps
        
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Use final timestep T for one-step generation
        t = torch.full((z.shape[0],), self.max_timesteps, device=z.device, dtype=torch.bfloat16)
        v = self.mmdit(z, t, y=c)  # Predict velocity field
        return z - v  # Apply velocity to denoise

class ApproximatedR1Regularization:
    """Approximated R1 regularization (Section 3.4)."""
    def __init__(self, sigma: float = 0.01, lambda_r1: float = 100.0):
        self.sigma = sigma
        self.lambda_r1 = lambda_r1
        
    def __call__(self, discriminator: APTDiscriminator, x: torch.Tensor, c: torch.Tensor, is_video: bool) -> torch.Tensor:
        x_perturbed = x + torch.randn_like(x, dtype=torch.bfloat16) * self.sigma
        d_real = discriminator(x, c, is_video)
        d_perturbed = discriminator(x_perturbed, c, is_video)
        return self.lambda_r1 * torch.mean((d_real - d_perturbed) ** 2)
    


class EMAModel:
    """Exponential Moving Average model tracking."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from mmditx import MMDiTX, PatchEmbed, TimestepEmbedder, VectorEmbedder
import os
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

# Existing supporting classes (abridged for brevity)
class APTCrossAttentionBlock(nn.Module):
    """Cross-attention block for APT discriminator (Section 3.3)."""
    def __init__(self, hidden_size: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.learnable_token = nn.Parameter(torch.randn(1, 1, hidden_size, dtype=dtype, device=device))
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype, device=device)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, dtype=dtype, device=device)
        self.proj = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        token = self.learnable_token.expand(B, -1, -1)
        x, token = self.norm(x), self.norm(token)
        q = self.qkv(token)[:, :, :self.hidden_size]
        k, v = self.qkv(x)[:, :, self.hidden_size:2*self.hidden_size], self.qkv(x)[:, :, 2*self.hidden_size:]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.hidden_size)
        return self.proj(x)

class APTDiscriminator(nn.Module):
    """Discriminator using MMDiT backbone (Section 3.3)."""
    def __init__(self, mmdit_model: MMDiTX, cross_attn_layers=[16, 26, 36], max_timesteps=1000, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.mmdit = mmdit_model
        self.hidden_size = mmdit_model.x_embedder.proj.out_channels  # Adjusted for MMDiTX
        self.num_heads = mmdit_model.num_heads
        self.max_timesteps = max_timesteps
        self.cross_attn_blocks = nn.ModuleList([APTCrossAttentionBlock(self.hidden_size, self.num_heads, device, dtype) for _ in cross_attn_layers])
        self.cross_attn_layers = cross_attn_layers
        self.final_norm = nn.LayerNorm(self.hidden_size * len(cross_attn_layers), dtype=dtype, device=device)
        self.final_proj = nn.Linear(self.hidden_size * len(cross_attn_layers), 1, dtype=dtype, device=device)

    def _shift_timestep(self, t: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * t / (1 + (shift - 1) * t)

    def _get_features(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.mmdit.x_embedder(x.squeeze(1) if x.dim() == 5 else x)
        c_mod = self.mmdit.t_embedder(t, dtype=x.dtype)
        if c is not None and hasattr(self.mmdit, 'y_embedder'):
            y = self.mmdit.y_embedder(c)
            c_mod = c_mod + y
        context = self.mmdit.context_embedder(c) if c is not None else None
        for i, block in enumerate(self.mmdit.joint_blocks):
            context, x = block(context, x, c=c_mod)
            if i in self.cross_attn_layers:
                features.append(x)
        return features

    def forward(self, x: torch.Tensor, c: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        t = torch.rand(x.shape[0], device=x.device) * self.max_timesteps
        shift = 12.0 if is_video else 1.0
        t = self._shift_timestep(t, shift)
        features = self._get_features(x, c, t)
        cross_attn_outputs = [block(feat) for feat, block in zip(features, self.cross_attn_blocks)]
        combined = torch.cat(cross_attn_outputs, dim=-1)
        return self.final_proj(self.final_norm(combined))

class APTGenerator(nn.Module):
    """One-step generator from distilled MMDiT (Section 3.2)."""
    def __init__(self, mmdit_model: MMDiTX, max_timesteps: int = 1000):
        super().__init__()
        self.mmdit = mmdit_model
        self.max_timesteps = max_timesteps

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = torch.full((z.shape[0],), self.max_timesteps, device=z.device, dtype=torch.bfloat16)
        v = self.mmdit(z, t, y=c)
        return z - v

class ApproximatedR1Regularization:
    """Approximated R1 regularization (Section 3.4)."""
    def __init__(self, sigma: float = 0.01, lambda_r1: float = 100.0):
        self.sigma = sigma
        self.lambda_r1 = lambda_r1

    def __call__(self, discriminator: APTDiscriminator, x: torch.Tensor, c: torch.Tensor, is_video: bool) -> torch.Tensor:
        x_perturbed = x + torch.randn_like(x, dtype=torch.bfloat16) * self.sigma
        d_real = discriminator(x, c, is_video)
        d_perturbed = discriminator(x_perturbed, c, is_video)
        return self.lambda_r1 * torch.mean((d_real - d_perturbed) ** 2)

class EMAModel:
    """EMA for generator stability (Section 3.5)."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model, self.decay = model, decay
        self.shadow = {name: param.data.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name], param.data = param.data, self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()

# Enhanced APTTrainer
class APTTrainer:
    """Trainer for Adversarial Post-Training (APT) with distributed support (Sections 3.1, 3.5)."""
    def __init__(
        self,
        generator: APTGenerator,
        discriminator: APTDiscriminator,
        r1_reg: ApproximatedR1Regularization,
        learning_rate: float,
        config: 'APTConfig',
        ema_decay: float = 0.995,
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = config.dtype

        # Move models to device and cast to dtype
        self.generator = generator.to(self.device, dtype=self.dtype)
        self.discriminator = discriminator.to(self.device, dtype=self.dtype)
        self.r1_reg = r1_reg

        # Optimizers with RMSProp as per Section 3.5
        self.g_optimizer = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=learning_rate,
            alpha=0.9,  # Equivalent to Adam beta2=0.9
            eps=1e-8,
        )
        self.d_optimizer = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=learning_rate,
            alpha=0.9,
            eps=1e-8,
        )

        # EMA for generator stability
        self.ema = EMAModel(self.generator, decay=ema_decay)

        # Distributed training setup
        self.is_distributed = config.world_size > 1
        if self.is_distributed:
            self.generator = DDP(self.generator, device_ids=[config.local_rank])
            self.discriminator = DDP(self.discriminator, device_ids=[config.local_rank])

    def set_learning_rate(self, lr: float):
        """Update learning rate dynamically (e.g., for phase transitions)."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr

    def set_r1_sigma(self, sigma: float):
        """Update R1 regularization sigma (e.g., image vs. video phase)."""
        self.r1_reg.sigma = sigma

    def train_step(
        self,
        real_samples: torch.Tensor,
        conditions: torch.Tensor,
        is_video: bool = False,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """Single training step with detailed loss computation (Section 3.1)."""
        real_samples = real_samples.to(self.device, dtype=self.dtype)
        conditions = conditions.to(self.device, dtype=self.dtype)

        # Generate noise if not provided
        if noise is None:
            shape = real_samples.shape  # [B, T, C, H, W] where T=1 for images, 48 for videos
            noise = torch.randn(shape, device=self.device, dtype=self.dtype)

        # Discriminator step
        self.d_optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=self.dtype):
            fake_samples = self.generator(noise, conditions)
            d_real = self.discriminator(real_samples, conditions, is_video)
            d_fake = self.discriminator(fake_samples.detach(), conditions, is_video)
            d_loss = -(torch.mean(F.logsigmoid(d_real)) + torch.mean(F.logsigmoid(-d_fake)))
            r1_loss = self.r1_reg(self.discriminator, real_samples, conditions, is_video)
            d_total_loss = d_loss + r1_loss

        d_total_loss.backward()
        if self.is_distributed:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()

        # Generator step
        self.g_optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=self.dtype):
            fake_samples = self.generator(noise, conditions)
            d_fake = self.discriminator(fake_samples, conditions, is_video)
            g_loss = -torch.mean(F.logsigmoid(d_fake))

        g_loss.backward()
        if self.is_distributed:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()
        self.ema.update()

        # Synchronize losses across GPUs if distributed
        if self.is_distributed:
            losses = [g_loss.clone(), d_loss.clone(), r1_loss.clone()]
            for loss in losses:
                dist.all_reduce(loss)
                loss /= self.config.world_size
            g_loss, d_loss, r1_loss = [loss.item() for loss in losses]
        else:
            g_loss, d_loss, r1_loss = g_loss.item(), d_loss.item(), r1_loss.item()

        return g_loss, d_loss, r1_loss

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        use_ema: bool = True,
        is_video: bool = False,
    ) -> torch.Tensor:
        """Generate samples with proper shape handling (Section 3.5)."""
        conditions = conditions.to(self.device, dtype=self.dtype)
        if use_ema:
            self.ema.apply_shadow()

        if noise is None:
            # Adjust shape based on image or video mode
            h, w = (64, 64) if not is_video else (45, 80)  # Latent sizes from paper
            t = 1 if not is_video else 48  # 1 frame for images, 48 for 2s@24fps videos
            shape = (conditions.shape[0], t, self.generator.mmdit.in_channels, h, w)
            noise = torch.randn(shape, device=self.device, dtype=self.dtype)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            samples = self.generator(noise, conditions)

        if use_ema:
            self.ema.restore()
        return samples

# Supporting classes from your addition
class APTConfig:
    def __init__(
        self,
        train_images: bool = True,
        train_videos: bool = True,
        image_size: int = 1024,
        video_width: int = 1280,
        video_height: int = 720,
        video_frames: int = 48,
        image_batch_size: int = 9062,
        video_batch_size: int = 2048,
        image_learning_rate: float = 5e-6,
        video_learning_rate: float = 3e-6,
        ema_decay: float = 0.995,
        image_r1_sigma: float = 0.01,
        video_r1_sigma: float = 0.1,
        r1_lambda: float = 100.0,
        image_updates: int = 350,
        video_updates: int = 300,
        world_size: int = 1,
        local_rank: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        log_interval: int = 10,
        save_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
    ):
        self.train_images = train_images
        self.train_videos = train_videos
        self.image_size, self.video_width, self.video_height, self.video_frames = image_size, video_width, video_height, video_frames
        self.image_batch_size, self.video_batch_size = image_batch_size, video_batch_size
        self.image_learning_rate, self.video_learning_rate = image_learning_rate, video_learning_rate
        self.ema_decay = ema_decay
        self.image_r1_sigma, self.video_r1_sigma, self.r1_lambda = image_r1_sigma, video_r1_sigma, r1_lambda
        self.image_updates, self.video_updates = image_updates, video_updates
        self.world_size, self.local_rank = world_size, local_rank
        self.device, self.dtype = device, dtype
        self.log_interval, self.save_interval, self.checkpoint_dir = log_interval, save_interval, checkpoint_dir
        if self.local_rank == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

class APTLogger:
    def __init__(self, config: APTConfig):
        self.config = config
        self.reset_metrics()
        if config.local_rank == 0:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    def reset_metrics(self):
        self.metrics = {"g_loss": [], "d_loss": [], "r1_loss": []}

    def log_metrics(self, g_loss: float, d_loss: float, r1_loss: float, step: int, phase: str):
        self.metrics["g_loss"].append(g_loss)
        self.metrics["d_loss"].append(d_loss)
        self.metrics["r1_loss"].append(r1_loss)
        if self.config.local_rank == 0 and step % self.config.log_interval == 0:
            avg_g_loss = np.mean(self.metrics["g_loss"][-self.config.log_interval:])
            avg_d_loss = np.mean(self.metrics["d_loss"][-self.config.log_interval:])
            avg_r1_loss = np.mean(self.metrics["r1_loss"][-self.config.log_interval:])
            logging.info(f"{phase} Step {step}: G_loss = {avg_g_loss:.4f}, D_loss = {avg_d_loss:.4f}, R1_loss = {avg_r1_loss:.4f}")

class APTCheckpointer:
    def __init__(self, config: APTConfig):
        self.config = config

    def save_checkpoint(self, trainer: 'APTTrainer', step: int, phase: str, is_final: bool = False):
        if self.config.local_rank == 0:
            checkpoint = {
                'generator_state_dict': trainer.generator.module.state_dict() if trainer.is_distributed else trainer.generator.state_dict(),
                'discriminator_state_dict': trainer.discriminator.module.state_dict() if trainer.is_distributed else trainer.discriminator.state_dict(),
                'g_optimizer_state_dict': trainer.g_optimizer.state_dict(),
                'd_optimizer_state_dict': trainer.d_optimizer.state_dict(),
                'ema_shadow': trainer.ema.shadow,
                'step': step,
                'phase': phase,
            }
            suffix = "final" if is_final else f"step_{step}"
            path = os.path.join(self.config.checkpoint_dir, f"apt_{phase}_{suffix}.pt")
            torch.save(checkpoint, path)
            logging.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, trainer: 'APTTrainer', path: str) -> Tuple[int, str]:
        checkpoint = torch.load(path, map_location=self.config.device)
        if trainer.is_distributed:
            trainer.generator.module.load_state_dict(checkpoint['generator_state_dict'])
            trainer.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
            trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        trainer.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        trainer.ema.shadow = checkpoint['ema_shadow']
        return checkpoint['step'], checkpoint['phase']

class APTTrainingLoop:
    def __init__(self, config: APTConfig, trainer: APTTrainer, image_dataloader=None, video_dataloader=None):
        self.config = config
        self.trainer = trainer
        self.image_dataloader = image_dataloader
        self.video_dataloader = video_dataloader
        self.logger = APTLogger(config)
        self.checkpointer = APTCheckpointer(config)
        if config.world_size > 1:
            dist.init_process_group("nccl")
            torch.cuda.set_device(config.local_rank)

    def train_images(self):
        if not self.config.train_images or self.image_dataloader is None:
            return
        self.logger.reset_metrics()
        self.trainer.set_learning_rate(self.config.image_learning_rate)
        self.trainer.set_r1_sigma(self.config.image_r1_sigma)
        iterator = iter(self.image_dataloader)
        for step in tqdm(range(self.config.image_updates), disable=self.config.local_rank != 0):
            try:
                real_images, conditions = next(iterator)
            except StopIteration:
                iterator = iter(self.image_dataloader)
                real_images, conditions = next(iterator)
            real_images = real_images.unsqueeze(1)  # [B, 1, C, H, W]
            g_loss, d_loss, r1_loss = self.trainer.train_step(real_images, conditions, is_video=False)
            self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "Image")
            if (step + 1) % self.config.save_interval == 0:
                self.checkpointer.save_checkpoint(self.trainer, step, "image")
        self.checkpointer.save_checkpoint(self.trainer, self.config.image_updates, "image", is_final=True)

    def train_videos(self):
        if not self.config.train_videos or self.video_dataloader is None:
            return
        self.logger.reset_metrics()
        self.trainer.set_learning_rate(self.config.video_learning_rate)
        self.trainer.set_r1_sigma(self.config.video_r1_sigma)
        iterator = iter(self.video_dataloader)
        for step in tqdm(range(self.config.video_updates), disable=self.config.local_rank != 0):
            try:
                real_videos, conditions = next(iterator)
            except StopIteration:
                iterator = iter(self.video_dataloader)
                real_videos, conditions = next(iterator)
            g_loss, d_loss, r1_loss = self.trainer.train_step(real_videos, conditions, is_video=True)
            self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "Video")
            if (step + 1) % self.config.save_interval == 0:
                self.checkpointer.save_checkpoint(self.trainer, step, "video")
        self.checkpointer.save_checkpoint(self.trainer, self.config.video_updates, "video", is_final=True)

    def train(self):
        self.train_images()
        self.train_videos()

def setup_apt_training(
    mmdit_model: nn.Module,
    config: APTConfig,
    image_dataloader: Optional[DataLoader] = None,
    video_dataloader: Optional[DataLoader] = None
) -> APTTrainingLoop:
    """Setup APT training components."""
    
    # Initialize generator and discriminator
    generator = APTGenerator(mmdit_model)
    discriminator = APTDiscriminator(
        mmdit_model,
        device=config.device,
        dtype=config.dtype
    )
    
    # Initialize R1 regularization
    r1_reg = ApproximatedR1Regularization(
        sigma=config.image_r1_sigma,
        lambda_r1=config.r1_lambda
    )
    
    # Create trainer
    trainer = APTTrainer(
        generator=generator,
        discriminator=discriminator,
        r1_reg=r1_reg,
        learning_rate=config.image_learning_rate,
        ema_decay=config.ema_decay
    )
    
    # Create training loop
    training_loop = APTTrainingLoop(
        config=config,
        trainer=trainer,
        image_dataloader=image_dataloader,
        video_dataloader=video_dataloader
    )
    
    return training_loop


# Example usage
if __name__ == "__main__":
    # Configuration
    config = APTConfig(
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )

    # Placeholder MMDiT model (pre-trained assumed)
    mmdit_model = MMDiTX(
        input_size=None,
        patch_size=2,
        in_channels=4,
        depth=36,
        mlp_ratio=4.0,
        learn_sigma=False,
        adm_in_channels=768,
        context_embedder_config={"target": "torch.nn.Linear", "params": {"in_features": 768, "out_features": 1152}},
        out_channels=4,
        pos_embed_max_size=64,
        num_patches=4096,
        dtype=config.dtype,
        device=config.device,
        verbose=True,
    )
    # Adjust hidden_size as before (placeholder adjustment)
    mmdit_model.x_embedder.proj = nn.Conv2d(4, 1152, kernel_size=2, stride=2, bias=True, dtype=config.dtype, device=config.device)
    for block in mmdit_model.joint_blocks:
        block.context_block.attn.qkv = nn.Linear(1152, 1152 * 3, bias=True, dtype=config.dtype, device=config.device)
        block.context_block.attn.proj = nn.Linear(1152, 1152, dtype=config.dtype, device=config.device)
        block.x_block.attn.qkv = nn.Linear(1152, 1152 * 3, bias=True, dtype=config.dtype, device=config.device)
        block.x_block.attn.proj = nn.Linear(1152, 1152, dtype=config.dtype, device=config.device)
        block.x_block.mlp.fc1 = nn.Linear(1152, int(1152 * 4), dtype=config.dtype, device=config.device)
        block.x_block.mlp.fc2 = nn.Linear(int(1152 * 4), 1152, dtype=config.dtype, device=config.device)
    mmdit_model.final_layer.linear = nn.Linear(1152, 2 * 2 * 4, bias=True, dtype=config.dtype, device=config.device)
    mmdit_model.final_layer.adaLN_modulation[-1] = nn.Linear(1152, 2 * 1152, bias=True, dtype=config.dtype, device=config.device)

    # Placeholder data loaders
    from torch.utils.data import Dataset, DataLoader
    class DummyDataset(Dataset):
        def __init__(self, is_video=False):
            self.is_video = is_video
        def __len__(self):
            return 1000
        def __getitem__(self, idx):
            if self.is_video:
                return torch.randn(48, 4, 45, 80), torch.randn(768)  # Video latents
            return torch.randn(4, 64, 64), torch.randn(768)  # Image latents

    image_loader = DataLoader(DummyDataset(is_video=False), batch_size=config.image_batch_size // config.world_size, shuffle=True)
    video_loader = DataLoader(DummyDataset(is_video=True), batch_size=config.video_batch_size // config.world_size, shuffle=True)

    # Setup and run training
    training_loop = setup_apt_training(mmdit_model, config, image_loader, video_loader)
    training_loop.train()

             
def initialize_models(pretrained_mmdit: MMDiTX, max_timesteps: int = 1000):
    """Initialize generator and discriminator (Sections 3.2, 3.3)."""
    # Generator from distilled weights (assumed pre-distilled here)
    generator = APTGenerator(pretrained_mmdit, max_timesteps)
    
    # Discriminator from original diffusion weights (deep copy to separate)
    import copy
    discriminator_mmdit = copy.deepcopy(pretrained_mmdit)
    discriminator = APTDiscriminator(discriminator_mmdit, max_timesteps=max_timesteps)
    
    return generator, discriminator

def train_apt(image_data_loader, video_data_loader):
    """Training pipeline with provided MMDiTX initialization (Section 3.5)."""
    # Initialize pre-trained MMDiTX as per paper's specs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_mmdit = MMDiTX(
        input_size=None,  # Dynamic size for latents (e.g., 64x64 for images, 45x80 for videos)
        patch_size=2,     # Paper implies latent patching (Section 3.1)
        in_channels=4,    # Typical VAE latent channels
        depth=36,         # 36 layers (Section 3.1)
        mlp_ratio=4.0,    # Standard DiT MLP ratio
        learn_sigma=False,  # Predict velocity, not noise (flow-matching, Section 3.1)
        adm_in_channels=768,  # Assuming CLIP-like text embeddings (adjustable)
        context_embedder_config={"target": "torch.nn.Linear", "params": {"in_features": 768, "out_features": 1152}},  # Text conditioning
        rmsnorm=False,    # Paper doesn't specify; use LayerNorm
        scale_mod_only=False,
        swiglu=False,     # Use GELU MLP (default in provided MMDiTX)
        out_channels=4,   # Match in_channels for velocity field
        pos_embed_max_size=64,  # Max latent size (e.g., 1024px / 16 = 64 patches)
        num_patches=4096,  # 64x64 patches for images
        qk_norm=None,     # No QK normalization specified
        x_block_self_attn_layers=[],  # No extra self-attn in paper
        qkv_bias=True,
        dtype=torch.bfloat16,  # BF16 per Section 3.5
        device=device,
        verbose=True
    )
    
    # Adjust hidden_size to aim for ~8B parameters (Section 3.1)
    # Default: hidden_size = 64 * depth = 64 * 36 = 2304; adjust to 1152 for realism
    pretrained_mmdit.x_embedder.proj = nn.Conv2d(4, 1152, kernel_size=2, stride=2, bias=True, dtype=torch.bfloat16, device=device)
    for block in pretrained_mmdit.joint_blocks:
        block.context_block.attn.qkv = nn.Linear(1152, 1152 * 3, bias=True, dtype=torch.bfloat16, device=device)
        block.context_block.attn.proj = nn.Linear(1152, 1152, dtype=torch.bfloat16, device=device)
        block.x_block.attn.qkv = nn.Linear(1152, 1152 * 3, bias=True, dtype=torch.bfloat16, device=device)
        block.x_block.attn.proj = nn.Linear(1152, 1152, dtype=torch.bfloat16, device=device)
        block.x_block.mlp.fc1 = nn.Linear(1152, int(1152 * 4), dtype=torch.bfloat16, device=device)
        block.x_block.mlp.fc2 = nn.Linear(int(1152 * 4), 1152, dtype=torch.bfloat16, device=device)
    pretrained_mmdit.final_layer.linear = nn.Linear(1152, 2 * 2 * 4, bias=True, dtype=torch.bfloat16, device=device)
    pretrained_mmdit.final_layer.adaLN_modulation[-1] = nn.Linear(1152, 2 * 1152, bias=True, dtype=torch.bfloat16, device=device)

    # Hypothetically load pre-trained weights (placeholder)
    # pretrained_mmdit.load_state_dict(torch.load("path_to_pretrained_weights.pth"), strict=False)

    # Image training phase
    generator, discriminator = initialize_models(pretrained_mmdit)
    r1_reg = ApproximatedR1Regularization(sigma=0.01, lambda_r1=100.0)
    trainer = APTTrainer(generator, discriminator, r1_reg, learning_rate=5e-6)
    for epoch in range(350):  # 350 updates (Section 3.5)
        for real_samples, conditions in image_data_loader:
            # Expected: real_samples [B, 4, 64, 64], conditions [B, 768]
            real_samples = real_samples.unsqueeze(1)  # Add time dim: [B, 1, 4, 64, 64]
            g_loss, d_loss = trainer.train_step(real_samples, conditions, is_video=False)
            print(f"Image Epoch {epoch}, G Loss: {g_loss}, D Loss: {d_loss}")

    # Save EMA checkpoint
    ema_checkpoint = trainer.ema.shadow.copy()

    # Video training phase
    generator, discriminator = initialize_models(pretrained_mmdit)
    for name, param in generator.named_parameters():
        if name in ema_checkpoint:
            param.data = ema_checkpoint[name].clone()
    r1_reg = ApproximatedR1Regularization(sigma=0.1, lambda_r1=100.0)
    trainer = APTTrainer(generator, discriminator, r1_reg, learning_rate=3e-6)
    for epoch in range(300):  # 300 updates (Section 3.5)
        for real_samples, conditions in video_data_loader:
            # Expected: real_samples [B, 48, 4, 45, 80], conditions [B, 768]
            g_loss, d_loss = trainer.train_step(real_samples, conditions, is_video=True)
            print(f"Video Epoch {epoch}, G Loss: {g_loss}, D Loss: {d_loss}")

    return trainer


# Example usage (placeholder for actual data and model)
# if __name__ == "__main__":
#     # Assume pretrained_mmdit is loaded (e.g., from a checkpoint)
#     pretrained_mmdit = MMDiTX(...)  # Replace with actual initialization
#     image_data_loader = ...  # Placeholder for 1024px image data loader (batch size 9062)
#     video_data_loader = ...  # Placeholder for 1280x720, 24fps, 2s video data loader (batch size 2048)
#     trainer = train_apt(image_data_loader, video_data_loader, pretrained_mmdit)

