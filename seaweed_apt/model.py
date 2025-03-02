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

