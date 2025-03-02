import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat

from mmditx import MMDiTX, DismantledBlock, PatchEmbed, TimestepEmbedder, VectorEmbedder

class LatentSpaceEncoder(nn.Module):
    """3D Causal VAE for compressing video and image inputs into latent space."""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,
        hidden_dims: List[int] = [64, 128, 256, 512],
        spatial_dims: int = 3,  # 3D for video, 2D for images
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Build encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, h_dim, 3, stride=2, padding=1, device=device, dtype=dtype)
                    if spatial_dims == 3
                    else nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1, device=device, dtype=dtype),
                    nn.BatchNorm3d(h_dim) if spatial_dims == 3 else nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim, device=device, dtype=dtype)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim, device=device, dtype=dtype)
        
        # Build decoder
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i], hidden_dims[i + 1], 3, 2, 1, 1, device=device, dtype=dtype)
                    if spatial_dims == 3
                    else nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], 3, 2, 1, 1, device=device, dtype=dtype),
                    nn.BatchNorm3d(hidden_dims[i + 1])
                    if spatial_dims == 3
                    else nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1], hidden_dims[-1], 3, 2, 1, 1, device=device, dtype=dtype)
            if spatial_dims == 3
            else nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 3, 2, 1, 1, device=device, dtype=dtype),
            nn.BatchNorm3d(hidden_dims[-1])
            if spatial_dims == 3
            else nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], 3, 3, 1, 1, device=device, dtype=dtype)
            if spatial_dims == 3
            else nn.Conv2d(hidden_dims[-1], 3, 3, 1, 1, device=device, dtype=dtype),
            nn.Tanh()
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent space."""
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        result = self.decoder(z)
        result = self.final_layer(result)
        return result
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var



class PoseGuider(nn.Module):
    """Lightweight convolutional encoder for pose heatmap sequences."""
    def __init__(
        self,
        in_channels: int = 33,  # MediaPipe's 33 keypoints
        out_dim: int = 1024,   # Match MMDiT model_dim
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=1, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv3d(128, out_dim // 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, device=device, dtype=dtype),
            nn.ReLU(),
        )
        # Input heatmaps: [B, T, 33, 64, 64]
        # After conv layers: [B, T, out_dim//4, H', W'] where H'=16, W'=16 (64 -> 32 -> 16 with stride=2)
        self.fc = nn.Linear((out_dim // 4) * 16 * 16, out_dim, device=device, dtype=dtype)
        
    def forward(self, pose_heatmaps: torch.Tensor) -> torch.Tensor:
        """Encode pose heatmaps into tokens."""
        # Input shape: [B, T, C, H, W] where C=33, H=W=64
        x = self.encoder(pose_heatmaps)  # [B, T, out_dim//4, 16, 16]
        x = rearrange(x, 'b t c h w -> b t (c h w)')  # Flatten spatial dims
        return self.fc(x)  # [B, T, out_dim]
    
class OmniConditionsModule(nn.Module):
    """Processes and combines multiple modality conditions for OmniHuman."""
    
    def __init__(
        self,
        model_dim: int,
        num_frames: int,
        audio_dim: int = 1024,
        pose_keypoints: int = 17,  # e.g., OpenPose 2D keypoints
        text_dim: int = 768,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Audio processor (unchanged)
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_dim, model_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, device=device, dtype=dtype)
        )
        
        # Pose guider for heatmap processing
        self.pose_guider = PoseGuider(
            in_channels=pose_keypoints,
            out_dim=model_dim,
            device=device,
            dtype=dtype
        )
        
        # Text processor 
        self.text_processor = VectorEmbedder(text_dim, model_dim, dtype=dtype, device=device)
        
        # Reference processor 
        self.ref_processor = PatchEmbed(
            img_size=None,
            patch_size=16,
            in_chans=3,
            embed_dim=model_dim,
            strict_img_size=False,
            dtype=dtype,
            device=device
        )
        
        # Temporal embeddings
        self.register_parameter(
            'temporal_embed',
            nn.Parameter(torch.randn(1, num_frames, model_dim, device=device, dtype=dtype))
        )
        
        # 3D RoPE embeddings
        self.rope_embeddings = nn.Parameter(
            torch.randn(1, num_frames, model_dim, device=device, dtype=dtype)
        )

    def process_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_tokens = self.audio_processor(audio_features)
        audio_tokens = torch.cat([audio_tokens[:, :-1], audio_tokens[:, 1:]], dim=-1)
        return audio_tokens

    def process_pose(self, pose_heatmaps: torch.Tensor) -> torch.Tensor:
        """Process pose heatmap sequences with PoseGuider."""
        # Input: [B, T, C, H, W] where C is keypoints, H/W are heatmap dims
        pose_tokens = self.pose_guider(pose_heatmaps)  # [B, T, model_dim]
        
        # Concatenate adjacent frames for temporal continuity
        pose_tokens_padded = torch.cat([pose_tokens[:, :-1], pose_tokens[:, 1:]], dim=-1)  # [B, T-1, 2*model_dim]
        return pose_tokens  # Return original tokens for stacking with latents

    def process_reference(self, reference_image: torch.Tensor, motion_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        ref_tokens = self.ref_processor(reference_image)
        if motion_frames is not None:
            motion_tokens = self.ref_processor(motion_frames)
            ref_tokens = torch.cat([ref_tokens, motion_tokens], dim=1)
        ref_tokens = ref_tokens + self.rope_embeddings[:, :ref_tokens.shape[1]]
        return ref_tokens

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        motion_frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        condition_tokens = {}
        
        if audio is not None:
            condition_tokens['audio'] = self.process_audio(audio)
        if pose is not None:
            condition_tokens['pose'] = self.process_pose(pose)
        if text is not None:
            condition_tokens['text'] = self.text_processor(text)
        if reference is not None:
            condition_tokens['reference'] = self.process_reference(reference, motion_frames)
            
        batch_size = next(iter(condition_tokens.values())).shape[0] if condition_tokens else 1
        condition_tokens['temporal'] = repeat(self.temporal_embed, '1 t d -> b t d', b=batch_size)
        
        return condition_tokens

class OmniHuman(nn.Module):
    """Main OmniHuman model combining MMDiT with multi-modal conditioning."""
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        model_dim: int = 1024,
        num_frames: int = 48,
        num_heads: int = 16,
        depth: int = 24,
        mlp_ratio: float = 4.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.vae = LatentSpaceEncoder(
            in_channels=in_channels,
            device=device,
            dtype=dtype
        )
        
        self.condition_processor = OmniConditionsModule(
            model_dim=model_dim,
            num_frames=num_frames,
            device=device,
            dtype=dtype
        )
        
        self.mmdit = MMDiTX(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels + model_dim,  # Add pose tokens to channel dim
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        motion_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encode input to latent space
        latent, mu, log_var = self.vae(x)  # [B, T, C_latent, H_latent, W_latent]
        
        # Process conditions
        conditions = self.condition_processor(
            audio=audio,
            pose=pose,
            text=text,
            reference=reference,
            motion_frames=motion_frames
        )
        
        # Stack pose tokens with noisy latents along channel dimension
        if 'pose' in conditions:
            pose_tokens = conditions['pose']  # [B, T, model_dim]
            pose_tokens = rearrange(pose_tokens, 'b t d -> b t d 1 1')  # Match latent spatial dims
            latent = torch.cat([latent, pose_tokens], dim=2)  # Stack along channel dimension
        
        # Denoise with MMDiT
        denoised = self.mmdit(
            latent,
            timesteps,
            context=conditions.get('temporal'),
            y=conditions.get('text'),
            controlnet_hidden_states=conditions.get('reference')  # Use reference as control
        )
        
        # Decode back to pixel space
        output = self.vae.decode(denoised)
        return output