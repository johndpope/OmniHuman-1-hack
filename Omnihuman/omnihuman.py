import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat

# Import Wan components
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

# Import Seaweed-Wan components (from our implementation)
from seaweed_apt.models import WanAPTGenerator

class PoseGuider(nn.Module):
    """Lightweight convolutional encoder for pose heatmap sequences."""
    def __init__(
        self,
        in_channels: int = 33,  # MediaPipe's 33 keypoints
        out_dim: int = 5120,    # Match Wan T2V-14B dim
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

class AudioProcessor(nn.Module):
    """Processes audio features to tokens suitable for conditioning."""
    def __init__(
        self,
        audio_dim: int = 1024,
        model_dim: int = 5120,  # Match Wan T2V-14B dim
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(audio_dim, model_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, device=device, dtype=dtype)
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Process audio features to model dimension tokens.
        
        Args:
            audio_features: Tensor of shape [B, T, audio_dim]
            
        Returns:
            Processed audio tokens of shape [B, T, model_dim]
        """
        audio_tokens = self.processor(audio_features)
        
        # Concatenate adjacent frames for temporal context
        if audio_tokens.shape[1] > 1:  # Ensure multiple frames exist
            audio_tokens_padded = torch.cat([
                audio_tokens[:, :-1], 
                audio_tokens[:, 1:]
            ], dim=-1)  # [B, T-1, 2*model_dim]
            return audio_tokens_padded
        
        return audio_tokens

class OmniConditionsModule(nn.Module):
    """Processes and combines multiple modality conditions for OmniHuman with Wan."""
    
    def __init__(
        self,
        model_dim: int = 5120,  # Match Wan T2V-14B dim
        num_frames: int = 49,   # 2 seconds at 24fps
        audio_dim: int = 1024,
        pose_keypoints: int = 33,  # MediaPipe keypoints
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            audio_dim=audio_dim,
            model_dim=model_dim,
            device=device,
            dtype=dtype
        )
        
        # Pose guider for heatmap processing
        self.pose_guider = PoseGuider(
            in_channels=pose_keypoints,
            out_dim=model_dim,
            device=device,
            dtype=dtype
        )
        
        # For reference images, we'll use Wan's VAE encoder
        # The actual processing will be done in forward method
        
        # Temporal embeddings (for sequential context)
        self.register_parameter(
            'temporal_embed',
            nn.Parameter(torch.randn(1, num_frames, model_dim, device=device, dtype=dtype) / (model_dim ** 0.5))
        )
        
        # Projection for combining conditions
        self.condition_projector = nn.Linear(model_dim, model_dim, device=device, dtype=dtype)

    def process_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Process audio features to suitable conditioning tokens."""
        return self.audio_processor(audio_features)

    def process_pose(self, pose_heatmaps: torch.Tensor) -> torch.Tensor:
        """Process pose heatmap sequences with PoseGuider."""
        # Input: [B, T, C, H, W] where C is keypoints, H/W are heatmap dims
        return self.pose_guider(pose_heatmaps)  # [B, T, model_dim]

    def process_reference(self, reference_image: torch.Tensor, vae: nn.Module) -> torch.Tensor:
        """Process reference image through VAE encoder."""
        # This will be called from the main OmniHuman model
        with torch.no_grad():
            reference_latent = vae.encode([reference_image])[0]
        return reference_latent

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process and combine all conditioning inputs.
        
        Args:
            audio: Audio features of shape [B, T, audio_dim]
            pose: Pose heatmaps of shape [B, T, keypoints, H, W]
            text_embeddings: Text embeddings from T5 encoder [B, L, text_dim]
            reference_latent: Encoded reference image from VAE [B, C, H, W]
            
        Returns:
            Dictionary of processed condition tokens
        """
        condition_tokens = {}
        
        if audio is not None:
            condition_tokens['audio'] = self.process_audio(audio)
        
        if pose is not None:
            condition_tokens['pose'] = self.process_pose(pose)
        
        if text_embeddings is not None:
            condition_tokens['text'] = text_embeddings
        
        if reference_latent is not None:
            condition_tokens['reference'] = reference_latent
            
        # Get batch size from any available tensor
        batch_size = next(iter(condition_tokens.values())).shape[0] if condition_tokens else 1
        
        # Add temporal embeddings
        condition_tokens['temporal'] = repeat(self.temporal_embed, '1 t d -> b t d', b=batch_size)
        
        return condition_tokens


class OmniHumanSeaweedWan(nn.Module):
    """OmniHuman model using Seaweed-Wan as the backbone."""
    def __init__(
        self,
        wan_config,
        checkpoint_dir,
        seaweed_checkpoint_path,
        num_frames: int = 49,  # 2 seconds at 24fps + 1 frame
        device_id: int = 0,
    ):
        super().__init__()
        self.device = torch.device(f"cuda:{device_id}")
        self.num_frames = num_frames
        self.config = wan_config
        
        # Initialize T5 text encoder from Wan
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=wan_config.t5_dtype,
            device=torch.device('cpu'),  # Load on CPU first
            checkpoint_path=f"{checkpoint_dir}/{wan_config.t5_checkpoint}",
            tokenizer_path=f"{checkpoint_dir}/{wan_config.t5_tokenizer}",
        )
        
        # Initialize VAE from Wan
        self.vae = WanVAE(
            vae_pth=f"{checkpoint_dir}/{wan_config.vae_checkpoint}",
            device=self.device
        )
        
        # Load the trained Seaweed-Wan model
        # This is the one-step generator initialized from consistency distillation
        # and trained with adversarial post-training
        original_model = WanModel.from_pretrained(checkpoint_dir)
        self.generator = WanAPTGenerator(original_model, final_timestep=wan_config.num_train_timesteps)
        self.generator.load_state_dict(torch.load(seaweed_checkpoint_path, map_location='cpu'))
        self.generator.to(self.device)
        self.generator.eval()
        
        # Initialize OmniConditions module
        self.condition_processor = OmniConditionsModule(
            model_dim=wan_config.dim,
            num_frames=num_frames,
            device=self.device
        )
        
        # Additional fusion layers for combining conditions
        self.condition_fusion = nn.Sequential(
            nn.Linear(wan_config.dim, wan_config.dim),
            nn.SiLU(),
            nn.Linear(wan_config.dim, wan_config.dim)
        )
        
    def forward(
        self,
        text_prompt: str,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        reference_image: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        seed: int = -1,
    ) -> torch.Tensor:
        """Generate a video using one-step Seaweed-Wan with multi-modal conditioning.
        
        Args:
            text_prompt: Text description for generation
            audio: Audio features [B, T, audio_dim]
            pose: Pose heatmaps [B, T, keypoints, H, W]
            reference_image: Reference image for appearance [B, 3, H, W]
            noise: Optional noise tensor for deterministic generation
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for noise generation
            
        Returns:
            Generated video tensor [B, 3, T, H, W]
        """
        # Process input conditions
        
        # Text processing
        self.text_encoder.model.to(self.device)
        text_context = self.text_encoder([text_prompt], self.device)
        text_context_null = self.text_encoder([self.config.sample_neg_prompt], self.device)
        self.text_encoder.model.cpu()  # Move back to CPU to save memory
        
        # Process reference image if provided
        reference_latent = None
        if reference_image is not None:
            reference_latent = self.condition_processor.process_reference(reference_image, self.vae)
        
        # Create a dictionary of condition tokens
        conditions = self.condition_processor(
            audio=audio,
            pose=pose,
            text_embeddings=text_context[0],
            reference_latent=reference_latent
        )
        
        # Calculate target shape for latent generation
        # This will depend on the VAE's compression ratio and desired output resolution
        target_width = 1280  # Example value
        target_height = 720  # Example value
        target_shape = (
            self.vae.model.z_dim, 
            self.num_frames,
            target_height // self.config.vae_stride[1],
            target_width // self.config.vae_stride[2]
        )
        
        # Generate noise if not provided
        if noise is None:
            if seed < 0:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            noise = torch.randn(
                target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=generator
            )
        
        # Calculate sequence length
        seq_len = (
            (target_shape[2] * target_shape[3]) // 
            (self.config.patch_size[1] * self.config.patch_size[2]) * 
            target_shape[1]
        )
        
        # Generate video latents with classifier-free guidance
        with torch.no_grad(), torch.amp.autocast(dtype=self.config.param_dtype):
            # Unconditional generation (using null text)
            uncond_latents = self.generator(
                [noise], 
                context=text_context_null,
                seq_len=seq_len
            )[0]
            
            # Conditional generation (using text and other conditions)
            cond_latents = self.generator(
                [noise], 
                context=text_context,
                seq_len=seq_len
            )[0]
            
            # Apply classifier-free guidance
            latents = uncond_latents + cfg_scale * (cond_latents - uncond_latents)
        
        # Decode latents to video
        with torch.no_grad():
            videos = self.vae.decode([latents])
        
        return videos[0]  # Return the generated video