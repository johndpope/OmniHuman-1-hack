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
    """Discriminator for APT using MMDiTX as base."""
    def __init__(
        self, 
        mmdit_model: MMDiTX,
        cross_attn_layers: List[int] = [16, 26, 36],
        device=None,
        dtype=None
    ):
        super().__init__()
        self.mmdit = mmdit_model
        self.hidden_size = mmdit_model.diffusion_model.hidden_size
        
        # Add cross-attention blocks at specified layers
        self.cross_attn_blocks = nn.ModuleList([
            APTCrossAttentionBlock(
                self.hidden_size,
                mmdit_model.num_heads,
                device=device,
                dtype=dtype
            ) for _ in cross_attn_layers
        ])
        self.cross_attn_layers = cross_attn_layers
        
        # Final projection layers
        self.final_norm = nn.LayerNorm(
            self.hidden_size * len(cross_attn_layers),
            dtype=dtype,
            device=device
        )
        self.final_proj = nn.Linear(
            self.hidden_size * len(cross_attn_layers),
            1,
            dtype=dtype,
            device=device
        )

    def _shift_timestep(self, t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
        """Apply timestep shifting as described in the paper."""
        return shift * t / (1 + (shift - 1) * t)

    def _get_features(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract features from intermediate layers of MMDiTX."""
        features = []
        context = None
        
        x = self.mmdit.x_embedder(x)
        c_mod = self.mmdit.t_embedder(t)
        
        if hasattr(self.mmdit, 'y_embedder') and c is not None:
            y = self.mmdit.y_embedder(c)
            c_mod = c_mod + y
            
        for i, block in enumerate(self.mmdit.joint_blocks):
            context, x = block(context, x, c=c_mod)
            if i in self.cross_attn_layers:
                features.append(x)
        
        return features

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        is_video: bool = False
    ) -> torch.Tensor:
        # Sample and shift timestep based on video/image mode
        t = torch.rand(x.shape[0], device=x.device)
        shift = 12.0 if is_video else 1.0
        t = self._shift_timestep(t, shift)
        
        # Get features from MMDiTX
        features = self._get_features(x, c, t)
        
        # Apply cross-attention blocks
        cross_attn_outputs = []
        for feature, block in zip(features, self.cross_attn_blocks):
            out = block(feature)
            cross_attn_outputs.append(out)
        
        # Combine outputs and project to scalar
        combined = torch.cat(cross_attn_outputs, dim=-1)
        combined = self.final_norm(combined)
        return self.final_proj(combined)

class APTGenerator(nn.Module):
    """One-step generator using MMDiTX."""
    def __init__(self, mmdit_model: MMDiTX):
        super().__init__()
        self.mmdit = mmdit_model
        
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Predict velocity field at final timestep (t=1)
        t = torch.ones(z.shape[0], device=z.device)
        v = self.mmdit(z, t, y=c)
        # Apply velocity field to get denoised sample
        return z - v

class ApproximatedR1Regularization:
    """Approximated R1 regularization for training stability."""
    def __init__(self, sigma: float = 0.01, lambda_r1: float = 100.0):
        self.sigma = sigma
        self.lambda_r1 = lambda_r1
        
    def __call__(
        self,
        discriminator: APTDiscriminator,
        x: torch.Tensor,
        c: torch.Tensor,
        is_video: bool = False
    ) -> torch.Tensor:
        # Add Gaussian noise to real samples
        x_perturbed = x + torch.randn_like(x) * self.sigma
        
        # Get discriminator outputs
        d_real = discriminator(x, c, is_video)
        d_perturbed = discriminator(x_perturbed, c, is_video)
        
        # Compute approximated R1 loss
        return self.lambda_r1 * torch.mean((d_real - d_perturbed) ** 2)

class EMAModel:
    """Exponential Moving Average model tracking."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.device = next(model.parameters()).device
        self.shadow = {}
        self.backup = {}
        
        # Register parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.shadow[name] * self.decay + 
                    param.data * (1 - self.decay)
                )
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

class APTTrainer:
    """Trainer for APT model."""
    def __init__(
        self,
        generator: APTGenerator,
        discriminator: APTDiscriminator,
        r1_reg: ApproximatedR1Regularization,
        learning_rate: float = 5e-6,
        ema_decay: float = 0.995
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.r1_reg = r1_reg
        
        # Setup optimizers (RMSprop with alpha=0.9)
        self.g_optimizer = torch.optim.RMSprop(
            generator.parameters(),
            lr=learning_rate,
            alpha=0.9
        )
        self.d_optimizer = torch.optim.RMSprop(
            discriminator.parameters(),
            lr=learning_rate,
            alpha=0.9
        )
        
        # Setup EMA for generator
        self.ema = EMAModel(generator, decay=ema_decay)
        
    def train_step(
        self,
        real_samples: torch.Tensor,
        conditions: torch.Tensor,
        is_video: bool = False,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[float, float]:
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(real_samples)
            
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake samples
        fake_samples = self.generator(noise, conditions)
        
        # Get discriminator outputs
        d_real = self.discriminator(real_samples, conditions, is_video)
        d_fake = self.discriminator(fake_samples.detach(), conditions, is_video)
        
        # Compute discriminator losses
        d_loss = -(torch.mean(F.logsigmoid(d_real)) + 
                  torch.mean(F.logsigmoid(-d_fake)))
        
        # Add R1 regularization
        r1_loss = self.r1_reg(
            self.discriminator,
            real_samples,
            conditions,
            is_video
        )
        
        d_total_loss = d_loss + r1_loss
        d_total_loss.backward()
        self.d_optimizer.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        # Recompute discriminator output for generated samples
        d_fake = self.discriminator(fake_samples, conditions, is_video)
        
        # Compute generator loss
        g_loss = -torch.mean(F.logsigmoid(d_fake))
        g_loss.backward()
        self.g_optimizer.step()
        
        # Update EMA model
        self.ema.update()
        
        return g_loss.item(), d_total_loss.item()

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        use_ema: bool = True
    ) -> torch.Tensor:
        """Generate samples using the trained generator."""
        if use_ema:
            self.ema.apply_shadow()
            
        if noise is None:
            noise = torch.randn(
                conditions.shape[0],
                *self.generator.mmdit.x_embedder.in_channels,
                device=conditions.device
            )
            
        samples = self.generator(noise, conditions)
        
        if use_ema:
            self.ema.restore()
            
        return samples
