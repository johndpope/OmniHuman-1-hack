import torch
import torch.nn as nn
import torch.nn.functional as F
from mmditx import MMDiTX, DismantledBlock, PatchEmbed, TimestepEmbedder, VectorEmbedder
from pylint.lint import Run
from logger import logger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open
from typing import *

class APTCrossAttentionBlock(nn.Module):
    """Modified Cross-attention block for SD3.5 architecture."""
    def __init__(self, hidden_size: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.learnable_token = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
        )
        
        # Modified for SD3.5 compatibility
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)
        self.proj = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        token = self.learnable_token.expand(B, -1, -1)
        
        x = self.norm(x)
        token = self.norm(token)
        
        q = self.to_q(token)
        k = self.to_k(x)
        v = self.to_v(x)
        
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.hidden_size)
        x = self.proj(x)
        
        return x

class APTDiscriminator(nn.Module):
    """Modified Discriminator for SD3.5 architecture."""
    def __init__(self, sd35_weights: dict, cross_attn_layers=[15, 25, 31], max_timesteps=1000, device="cuda", dtype=torch.bfloat16):        
        super().__init__()
        self.model = MMDiTX(
            input_size=None,
            patch_size=2,
            in_channels=16,
            depth=32,  # SD3.5 Medium confirmed
            mlp_ratio=4.0,
            learn_sigma=False,
            adm_in_channels=2048,  # Matches weights
            context_embedder_config={"target": "torch.nn.Linear", "params": {"in_features": 2048, "out_features": 1536}},
            out_channels=64,  # Matches weights
            pos_embed_max_size=128,  # Adjusted to accommodate larger inputs
            num_patches=4096,
            qk_norm=None,
            x_block_self_attn_layers=[],
            qkv_bias=True,
            dtype=dtype,
            device=device,
            verbose=True
        )
        
        # Get hidden size from model config or default to SD3.5 size
        self.hidden_size = 1536  # From weights
        self.num_heads = 24      # 1536 / 64 = 24
        self.max_timesteps = max_timesteps
        self.cross_attn_blocks = nn.ModuleList([
            APTCrossAttentionBlock(self.hidden_size, self.num_heads, device, dtype)
            for _ in cross_attn_layers
        ])
        self.cross_attn_layers = cross_attn_layers
        self.final_norm = nn.LayerNorm(self.hidden_size * len(cross_attn_layers), eps=1e-6, dtype=dtype, device=device)
        self.final_proj = nn.Linear(self.hidden_size * len(cross_attn_layers), 1, dtype=dtype, device=device)
        self.model.load_state_dict(sd35_weights, strict=False)
    
    def _shift_timestep(self, t: torch.Tensor, shift: float) -> torch.Tensor:
        """Timestep shifting with SD3.5 compatibility."""
        return shift * t / (1 + (shift - 1) * t)

    def _get_features(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        # Modified for SD3.5 feature extraction
        x = self.model.get_input_embeddings()(x)
        
        # Handle conditioning
        c_emb = self.model.get_text_embeddings(c) if c is not None else None
        t_emb = self.model.get_time_embeddings(t)
        
        # Combine embeddings
        hidden_states = x
        for i, block in enumerate(self.model.blocks):
            hidden_states = block(
                hidden_states,
                c_emb,
                t_emb
            )
            if i in self.cross_attn_layers:
                features.append(hidden_states)
                
        return features

    def forward(self, x: torch.Tensor, c: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        t = torch.rand(x.shape[0], device=x.device) * self.max_timesteps
        shift = 12.0 if is_video else 1.0
        t = self._shift_timestep(t, shift)
        features = []
        x = self.model.x_embedder(x)  # Simplified feature extraction
        c_emb = self.model.y_embedder(c) if c is not None else None
        t_emb = self.model.t_embedder(t, dtype=x.dtype)
        hidden_states = x
        for i, block in enumerate(self.model.joint_blocks):
            _, hidden_states = block(hidden_states, t_emb, c_emb)  # Assuming context is optional
            if i in self.cross_attn_layers:
                features.append(hidden_states)
        cross_attn_outputs = [block(feat) for feat, block in zip(features, self.cross_attn_blocks)]
        combined = torch.cat(cross_attn_outputs, dim=-1)
        return self.final_proj(self.final_norm(combined))
    

class APTGenerator(nn.Module):
    def __init__(self, sd35_weights: dict, max_timesteps: int = 1000, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        # Initialize MMDiT matching SD3.5 structure
        self.mmdit = MMDiTX(
            input_size=None,
            patch_size=2,
            in_channels=16,
            depth=32,  # SD3.5 Medium confirmed
            mlp_ratio=4.0,
            learn_sigma=False,
            adm_in_channels=2048,  # Matches weights
            context_embedder_config={"target": "torch.nn.Linear", "params": {"in_features": 2048, "out_features": 1536}},
            out_channels=64,  # Matches weights
            pos_embed_max_size=128,  # Adjusted to accommodate larger inputs
            num_patches=4096,
            qk_norm=None,
            x_block_self_attn_layers=[],
            qkv_bias=True,
            dtype=dtype,
            device=device,
            verbose=True
        )

        self.mmdit.load_state_dict(sd35_weights, strict=False)
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
            logger.info(f"{phase} Step {step}: G_loss = {avg_g_loss:.4f}, D_loss = {avg_d_loss:.4f}, R1_loss = {avg_r1_loss:.4f}")

class APTCheckpointer:
    def __init__(self, config: APTConfig):
        self.config = config

    def save_checkpoint(self, trainer: 'SeaweedTrainer', step: int, phase: str, is_final: bool = False):
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
            logger.info(f"Saved checkpoint to {path}")

def load_sd35_weights(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> dict:
    logger.info(f"Loading SD3.5 weights from {model_path}")
    with safe_open(model_path, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k).to(dtype=dtype) for k in f.keys()}
    
    logger.info(f"Loaded {len(state_dict)} weight tensors")
    expected_mappings = {
        "model.diffusion_model.x_embedder.proj.weight": (1536, 16, 2, 2),
        "model.diffusion_model.y_embedder.mlp.0.weight": (1536, 2048),
        "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight": (4608, 1536),
        "model.diffusion_model.final_layer.linear.weight": (64, 1536),
    }
    
    mapped_state_dict = {}
    prefix = "model.diffusion_model."
    missing_keys = []
    shape_mismatches = []
    
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            mapped_key = new_key
            if "x_embedder" in new_key:
                mapped_key = f"x_embedder.{new_key.split('x_embedder.')[-1]}"
            elif "y_embedder" in new_key:
                mapped_key = f"y_embedder.{new_key.split('y_embedder.')[-1]}"
            mapped_state_dict[mapped_key] = tensor
            if key in expected_mappings:
                expected_shape = expected_mappings[key]
                if tuple(tensor.shape) != expected_shape:
                    shape_mismatches.append(f"{key}: Expected {expected_shape}, got {tuple(tensor.shape)}")
            logger.debug(f"Loaded {key} -> {mapped_key}: {tensor.shape}")
        else:
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"Keys not mapped: {len(missing_keys)} (e.g., {missing_keys[:5]})")
    if shape_mismatches:
        logger.error(f"Shape mismatches detected: {shape_mismatches}")
        raise ValueError("Weight shape validation failed")
    logger.info("All key shapes validated successfully")
    return mapped_state_dict



    

class SeaweedTrainer:
    def __init__(
        self,
        model_path: str,
        config: APTConfig,
        image_dataloader: Optional[torch.utils.data.DataLoader] = None,
        video_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self.config = config
        self.device = 'cuda'
        self.dtype = config.dtype
        
        logger.info("Loading SD3.5 weights...")
        self.sd35_weights = load_sd35_weights(model_path, device=self.device, dtype=self.dtype)
        
        logger.info("Initializing generator with SD3.5 weights...")
        self.generator = APTGenerator(
            sd35_weights=self.sd35_weights,
            max_timesteps=1000,
            device=self.device,
            dtype=self.dtype
        ).to(self.device, dtype=self.dtype)
        
        logger.info("Initializing discriminator with SD3.5 weights...")
        self.discriminator = APTDiscriminator(
            sd35_weights=self.sd35_weights,
            cross_attn_layers=[15, 25, 31],
            max_timesteps=1000,
            device=self.device,
            dtype=self.dtype
        ).to(self.device, dtype=self.dtype)
        
        self.r1_reg = ApproximatedR1Regularization(sigma=self.config.image_r1_sigma, lambda_r1=self.config.r1_lambda)
        self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=self.config.image_learning_rate, alpha=0.9, eps=1e-8)
        self.d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.config.image_learning_rate, alpha=0.9, eps=1e-8)
        self.ema = EMAModel(self.generator, decay=self.config.ema_decay)
        
        self.is_distributed = self.config.world_size > 1
        if self.is_distributed:
            dist.init_process_group("nccl")
            torch.cuda.set_device(self.config.local_rank)
            self.generator = DDP(self.generator, device_ids=[self.config.local_rank])
            self.discriminator = DDP(self.discriminator, device_ids=[self.config.local_rank])
        
        self.logger = APTLogger(self.config)
        self.checkpointer = APTCheckpointer(self.config)
        self.image_dataloader = image_dataloader
        self.video_dataloader = video_dataloader
        self._validate_initialization()

    def _validate_initialization(self):
        batch_size = 2
        noise = torch.randn(batch_size, 1, 16, 64, 64, device=self.device, dtype=self.dtype)
        condition = torch.randn(batch_size, 2048, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            gen_output = self.generator(noise, condition)
            assert gen_output.shape == (batch_size, 1, 16, 64, 64), f"Generator output shape mismatch: {gen_output.shape}"
            disc_output = self.discriminator(noise, condition, is_video=False)
            assert disc_output.shape == (batch_size, 1), f"Discriminator output shape mismatch: {disc_output.shape}"
        logger.info("Initialization validated successfully")

    def set_learning_rate(self, lr: float):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr

    def set_r1_sigma(self, sigma: float):
        self.r1_reg.sigma = sigma

    def train_step(self, real_samples: torch.Tensor, conditions: torch.Tensor, is_video: bool = False, noise: Optional[torch.Tensor] = None) -> Dict[str, float]:
        real_samples = real_samples.to(self.device, dtype=self.dtype)
        conditions = conditions.to(self.device, dtype=self.dtype)
        if noise is None:
            noise = torch.randn_like(real_samples, device=self.device, dtype=self.dtype)
        
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
        
        if self.is_distributed:
            losses = [g_loss.clone(), d_loss.clone(), r1_loss.clone()]
            for loss in losses:
                dist.all_reduce(loss)
                loss /= self.config.world_size
            g_loss, d_loss, r1_loss = [loss.item() for loss in losses]
        else:
            g_loss, d_loss, r1_loss = g_loss.item(), d_loss.item(), r1_loss.item()
        
        return {"g_loss": g_loss, "d_loss": d_loss, "r1_loss": r1_loss}

    @torch.no_grad()
    def generate_samples(self, conditions: torch.Tensor, num_samples: int = 1, use_ema: bool = True, is_video: bool = False) -> torch.Tensor:
        if use_ema:
            self.ema.apply_shadow()
        
        conditions = conditions.to(self.device, dtype=self.dtype)
        h, w = (64, 64) if not is_video else (45, 80)
        t = 1 if not is_video else 48
        noise = torch.randn(num_samples, t, 16, h, w, device=self.device, dtype=self.dtype)
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            samples = self.generator(noise, conditions)
        
        if use_ema:
            self.ema.restore()
        return samples

    def train_images(self):
        if not self.config.train_images or self.image_dataloader is None:
            return
        self.logger.reset_metrics()
        self.set_learning_rate(self.config.image_learning_rate)
        self.set_r1_sigma(self.config.image_r1_sigma)
        iterator = iter(self.image_dataloader)
        
        for step in tqdm(range(self.config.image_updates), disable=self.config.local_rank != 0):
            try:
                real_images, conditions = next(iterator)
            except StopIteration:
                iterator = iter(self.image_dataloader)
                real_images, conditions = next(iterator)
            real_images = real_images.unsqueeze(1)
            g_loss, d_loss, r1_loss = self.train_step(real_images, conditions, is_video=False)
            self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "Image")
            if (step + 1) % self.config.save_interval == 0:
                self.checkpointer.save_checkpoint(self, step, "image")
        self.checkpointer.save_checkpoint(self, self.config.image_updates, "image", is_final=True)

    def train_videos(self):
        if not self.config.train_videos or self.video_dataloader is None:
            return
        self.logger.reset_metrics()
        self.set_learning_rate(self.config.video_learning_rate)
        self.set_r1_sigma(self.config.video_r1_sigma)
        iterator = iter(self.video_dataloader)
        
        for step in tqdm(range(self.config.video_updates), disable=self.config.local_rank != 0):
            try:
                real_videos, conditions = next(iterator)
            except StopIteration:
                iterator = iter(self.video_dataloader)
                real_videos, conditions = next(iterator)
            g_loss, d_loss, r1_loss = self.train_step(real_videos, conditions, is_video=True)
            self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "Video")
            if (step + 1) % self.config.save_interval == 0:
                self.checkpointer.save_checkpoint(self, step, "video")
        self.checkpointer.save_checkpoint(self, self.config.video_updates, "video", is_final=True)

    def train(self):
        self.train_images()
        self.train_videos()

def main():
    config = APTConfig(
        train_images=True,
        train_videos=True,
        image_batch_size=8,
        video_batch_size=8,
        world_size=1,
        local_rank=0,
        device="cuda",
        dtype=torch.bfloat16
    )
    
    class DummyDataset(Dataset):
        def __init__(self, is_video=False):
            self.is_video = is_video
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            if self.is_video:
                return torch.randn(48, 16, 45, 80), torch.randn(2048)
            return torch.randn(16, 64, 64), torch.randn(2048)
    
    image_loader = DataLoader(DummyDataset(is_video=False), batch_size=config.image_batch_size)
    video_loader = DataLoader(DummyDataset(is_video=True), batch_size=config.video_batch_size)
    
    trainer = SeaweedTrainer(
        model_path="./models/sd3.5_medium.safetensors",
        config=config,
        image_dataloader=image_loader,
        video_dataloader=video_loader
    )
    trainer.train()

if __name__ == "__main__":
    main()