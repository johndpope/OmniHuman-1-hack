import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat
from wan import WanT2V
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.configs import t2v_14B  # Example config, adjust as needed
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from logger import logger
from omegaconf import DictConfig, OmegaConf

class OmniConditionsModule(nn.Module):
    """Processes and combines multiple modality conditions for OmniHuman."""
    
    def __init__(
        self,
        model_dim: int = 5120,  # Match Wan T2V-14B dim
        num_frames: int = 49,   # 2 seconds at 24fps + 1 frame
        audio_dim: int = 1024,
        pose_keypoints: int = 33,  # MediaPipe keypoints
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Audio processor
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_dim, model_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, device=device, dtype=dtype)
        )
        
        # Pose guider
        self.pose_guider = nn.Sequential(
            nn.Conv3d(pose_keypoints, 64, kernel_size=(3, 3, 3), padding=1, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv3d(128, model_dim // 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, device=device, dtype=dtype),
            nn.ReLU(),
        )
        self.pose_fc = nn.Linear((model_dim // 4) * 16 * 16, model_dim, device=device, dtype=dtype)
        
        # Temporal embeddings
        self.temporal_embed = nn.Parameter(
            torch.randn(1, num_frames, model_dim, device=device, dtype=dtype) / (model_dim ** 0.5)
        )
        
        # Condition projector
        self.condition_projector = nn.Linear(model_dim, model_dim, device=device, dtype=dtype)

    def process_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Process audio features into tokens."""
        audio_tokens = self.audio_processor(audio_features)  # [B, T, model_dim]
        if audio_tokens.shape[1] > 1:
            audio_tokens = torch.cat([audio_tokens[:, :-1], audio_tokens[:, 1:]], dim=-1)  # [B, T-1, 2*model_dim]
        return audio_tokens

    def process_pose(self, pose_heatmaps: torch.Tensor) -> torch.Tensor:
        """Process pose heatmaps into tokens."""
        x = self.pose_guider(pose_heatmaps)  # [B, T, model_dim//4, 16, 16]
        x = rearrange(x, 'b t c h w -> b t (c h w)')
        return self.pose_fc(x)  # [B, T, model_dim]

    def process_reference(self, reference_image: torch.Tensor, vae: nn.Module) -> torch.Tensor:
        """Encode reference image using VAE."""
        with torch.no_grad():
            return vae.encode([reference_image])[0]  # [B, C, H_latent, W_latent]

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        condition_tokens = {}
        batch_size = next(iter(locals().values())).shape[0] if any(locals().values()) else 1
        
        if audio is not None:
            condition_tokens['audio'] = self.process_audio(audio)
        if pose is not None:
            condition_tokens['pose'] = self.process_pose(pose)
        if text_embeddings is not None:
            condition_tokens['text'] = text_embeddings
        if reference_latent is not None:
            condition_tokens['reference'] = reference_latent
        condition_tokens['temporal'] = repeat(self.temporal_embed, '1 t d -> b t d', b=batch_size)
        
        return condition_tokens

class OmniHumanWanT2V(nn.Module):
    """OmniHuman model using WanT2V with multi-step diffusion and Sapiens 308 keypoints."""
    
    def __init__(
        self,
        config: DictConfig,
        device_id: int = 0,
    ):
        """Initialize the OmniHuman model with the WanT2V backbone.
        
        Args:
            config: Configuration dictionary with model parameters
            device_id: CUDA device ID to use
        """
        super().__init__()
        self.config = config
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.num_frames = config.get("num_frames", 49)
        self.num_keypoints = config.get("num_keypoints", 308)  # Sapiens 308 keypoints
        
        # Initialize WanT2V
        self.wan_t2v = WanT2V(
            config=config.get("wan_config", t2v_14B),
            checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=config.get("t5_cpu", False)
        )
        self.wan_t2v.model.to(self.device)
        
        # Initialize condition processors
        self._init_condition_processors()
        
        # Initialize diffusion scheduler
        self._init_diffusion_scheduler()
        
        logger.info(f"Initialized OmniHumanWanT2V with {self.num_keypoints} keypoints")
        
    def _init_condition_processors(self):
        """Initialize processors for different condition modalities."""
        model_dim = self.config.get("model_dim", 5120)  # Match Wan T2V-14B dim
        
        # Audio processor
        self.audio_processor = nn.Sequential(
            nn.Linear(self.config.get("audio_dim", 1024), model_dim, device=self.device),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, device=self.device)
        )
        
        # Pose processor (for 308 keypoints from Sapiens)
        self.pose_processor = nn.Sequential(
            nn.Conv3d(self.num_keypoints, 128, kernel_size=(3, 3, 3), padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv3d(256, model_dim // 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, device=self.device),
            nn.ReLU(),
        )
        self.pose_fc = nn.Linear((model_dim // 4) * 16 * 16, model_dim, device=self.device)
        
        # Temporal embeddings
        self.temporal_embed = nn.Parameter(
            torch.randn(1, self.num_frames, model_dim, device=self.device) / (model_dim ** 0.5)
        )
        
        # Condition projector
        self.condition_projector = nn.Linear(model_dim, model_dim, device=self.device)
        
        logger.debug(f"Initialized condition processors with model_dim={model_dim}")
        
    def _init_diffusion_scheduler(self):
        """Initialize the diffusion scheduler."""
        # Import here to avoid circular dependencies
        try:
            from diffusers import DPMSolverMultistepScheduler
            self.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                solver_order=2,
                prediction_type="v_prediction",
                beta_schedule="linear",
                clip_sample=False
            )
            logger.info("Using DPMSolverMultistepScheduler from diffusers")
        except ImportError:
            # Fallback to custom scheduler if diffusers is not available
            from utils.fm_solvers import FlowDPMSolverMultistepScheduler
            self.scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                solver_order=2,
                prediction_type="flow_prediction",
                shift=1.0
            )
            logger.info("Using custom FlowDPMSolverMultistepScheduler")
    
    def process_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Process audio features into tokens.
        
        Args:
            audio_features: Audio features [B, T, audio_dim]
            
        Returns:
            Audio tokens [B, T, model_dim]
        """
        audio_tokens = self.audio_processor(audio_features)  # [B, T, model_dim]
        logger.debug(f"Audio tokens shape: {audio_tokens.shape}")
        
        if audio_tokens.shape[1] > 1:
            # Add temporal context by concatenating adjacent frames
            audio_tokens = torch.cat([
                audio_tokens[:, :-1], 
                audio_tokens[:, 1:]
            ], dim=-1)  # [B, T-1, 2*model_dim]
            logger.debug(f"Audio tokens with temporal context: {audio_tokens.shape}")
            
        return audio_tokens

    def process_pose(self, pose_heatmaps: torch.Tensor) -> torch.Tensor:
        """Process pose heatmaps into tokens.
        
        Args:
            pose_heatmaps: Pose heatmaps [B, T, num_keypoints, H, W]
            
        Returns:
            Pose tokens [B, T, model_dim]
        """
        logger.debug(f"Processing pose heatmaps: {pose_heatmaps.shape}")
        x = self.pose_processor(pose_heatmaps)  # [B, T, model_dim//4, 16, 16]
        logger.debug(f"Pose features shape after conv: {x.shape}")
        
        x = rearrange(x, 'b t c h w -> b t (c h w)')
        logger.debug(f"Pose features shape after rearrange: {x.shape}")
        
        tokens = self.pose_fc(x)  # [B, T, model_dim]
        logger.debug(f"Final pose tokens shape: {tokens.shape}")
        
        return tokens

    def process_reference(self, reference_image: torch.Tensor) -> torch.Tensor:
        """Encode reference image using VAE.
        
        Args:
            reference_image: Reference image [B, C, H, W]
            
        Returns:
            Reference latent [B, C, H_latent, W_latent]
        """
        logger.debug(f"Processing reference image: {reference_image.shape}")
        with torch.no_grad():
            latent = self.wan_t2v.vae.encode([reference_image])[0]
            logger.debug(f"Reference latent shape: {latent.shape}")
            return latent

    def prepare_conditions(
        self,
        text_prompt: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        reference_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare all conditioning inputs.
        
        Args:
            text_prompt: Text prompt for conditioning
            audio: Audio features [B, T, audio_dim]
            pose: Pose heatmaps [B, T, num_keypoints, H, W]
            reference_image: Reference image [B, C, H, W]
            
        Returns:
            Dictionary of condition tokens
        """
        condition_tokens = {}
        batch_size = 1  # Default batch size
        
        # Determine batch size from available inputs
        if audio is not None:
            batch_size = audio.shape[0]
        elif pose is not None:
            batch_size = pose.shape[0]
        elif reference_image is not None:
            batch_size = reference_image.shape[0]
            
        logger.debug(f"Preparing conditions with batch_size={batch_size}")
        
        # Process text prompt
        if text_prompt is not None:
            self.wan_t2v.text_encoder.model.to(self.device)
            text_embeddings = self.wan_t2v.text_encoder([text_prompt], self.device)[0]
            condition_tokens['text'] = text_embeddings
            self.wan_t2v.text_encoder.model.cpu()
            logger.debug(f"Text embeddings shape: {text_embeddings.shape}")
            
        # Process audio
        if audio is not None:
            condition_tokens['audio'] = self.process_audio(audio)
            
        # Process pose
        if pose is not None:
            condition_tokens['pose'] = self.process_pose(pose)
            
        # Process reference image
        if reference_image is not None:
            condition_tokens['reference'] = self.process_reference(reference_image)
            
        # Add temporal embeddings
        condition_tokens['temporal'] = repeat(
            self.temporal_embed, 
            '1 t d -> b t d', 
            b=batch_size
        )
        
        return condition_tokens

    def _compute_seq_len(self, shape: Tuple[int, ...]) -> int:
        """Compute sequence length for WanT2V based on latent shape.
        
        Args:
            shape: Shape of the latent [B, C, T, H, W]
            
        Returns:
            Sequence length for the model
        """
        patch_size = self.wan_t2v.model.patch_size
        return (shape[2] // patch_size[0]) * (shape[3] // patch_size[1]) * (shape[4] // patch_size[2])

    def forward(
        self,
        text_prompt: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        reference_image: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 7.5,
        seed: int = -1,
    ) -> torch.Tensor:
        """Generate a video using multi-step diffusion.
        
        Args:
            text_prompt: Text prompt for conditioning
            audio: Audio features [B, T, audio_dim]
            pose: Pose heatmaps [B, T, num_keypoints, H, W]
            reference_image: Reference image [B, C, H, W]
            num_inference_steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for generation
            
        Returns:
            Generated video [B, C, T, H, W]
        """
        batch_size = 1  # Single sample generation for simplicity
        logger.info(f"Generating video with {num_inference_steps} inference steps")
        
        # Process conditions
        conditions = self.prepare_conditions(
            text_prompt=text_prompt,
            audio=audio,
            pose=pose,
            reference_image=reference_image
        )
        
        # Get null text embeddings for classifier-free guidance
        if text_prompt is not None:
            self.wan_t2v.text_encoder.model.to(self.device)
            text_context = conditions['text']
            text_context_null = self.wan_t2v.text_encoder(
                [self.config.get("negative_prompt", "")], 
                self.device
            )[0]
            self.wan_t2v.text_encoder.model.cpu()
        else:
            text_context = None
            text_context_null = None
        
        # Calculate target latent shape based on reference image or default values
        if reference_image is not None:
            ref_latent = conditions.get('reference')
            h = ref_latent.shape[-2] if ref_latent is not None else 64
            w = ref_latent.shape[-1] if ref_latent is not None else 64
        else:
            h = self.config.get("latent_height", 64)
            w = self.config.get("latent_width", 64)
        
        target_shape = (
            batch_size,
            self.wan_t2v.vae.model.z_dim,
            self.num_frames,
            h,
            w
        )
        logger.debug(f"Target latent shape: {target_shape}")
        
        # Generate initial noise
        if seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.debug(f"Using seed: {seed}")
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        latents = torch.randn(
            target_shape, 
            device=self.device, 
            generator=generator, 
            dtype=torch.float32
        )
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Multi-step diffusion with CFG
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                logger.debug(f"Diffusion step {i+1}/{len(timesteps)}, t={t.item()}")
                
                # Prepare conditional and unconditional inputs
                noisy_latents = latents
                reference_latent = conditions.get('reference')
                
                if reference_latent is not None:
                    # Stack reference along temporal dim for self-attention
                    noisy_latents = torch.cat([reference_latent, noisy_latents], dim=2)
                    logger.debug(f"Latents with reference shape: {noisy_latents.shape}")
                
                # Calculate unconditional prediction if doing CFG
                uncond_pred = None
                if cfg_scale > 1.0 and text_context is not None:
                    uncond_pred = self.wan_t2v.model(
                        noisy_latents,
                        t.unsqueeze(0),
                        context=[text_context_null],
                        seq_len=self._compute_seq_len(target_shape)
                    )[0]
                    logger.debug(f"Unconditional prediction shape: {uncond_pred.shape}")
                
                # Conditional prediction
                context = [text_context] if text_context is not None else None
                cond_pred = self.wan_t2v.model(
                    noisy_latents,
                    t.unsqueeze(0),
                    context=context,
                    seq_len=self._compute_seq_len(target_shape),
                    extra_conditions=conditions
                )[0]
                logger.debug(f"Conditional prediction shape: {cond_pred.shape}")
                
                # Apply CFG if needed
                if cfg_scale > 1.0 and text_context is not None and uncond_pred is not None:
                    # CFG annealing for reducing wrinkles while maintaining expressiveness
                    # Linearly reduce CFG from initial value to 1.0
                    progress = i / len(timesteps)
                    current_cfg = cfg_scale * (1.0 - progress) + 1.0 * progress
                    logger.debug(f"Current CFG scale: {current_cfg}")
                    
                    latents_pred = uncond_pred + current_cfg * (cond_pred - uncond_pred)
                else:
                    latents_pred = cond_pred
                
                # Step with scheduler
                latents = self.scheduler.step(latents_pred, t, latents).prev_sample
                logger.debug(f"Updated latents shape: {latents.shape}")
        
        # Decode latents to video
        with torch.no_grad():
            video = self.wan_t2v.vae.decode([latents])[0]
            logger.info(f"Generated video shape: {video.shape}")
        
        return video

    def training_step(
        self,
        frames: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow-matching loss for training.
        
        Args:
            frames: Ground truth frames [B, C, T, H, W]
            conditions: Dictionary of condition tokens
            t: Timestep values [B]
            
        Returns:
            Flow matching loss
        """
        logger.debug(f"Training step with frames shape: {frames.shape}")
        
        # Add noise using flow-matching
        noise = torch.randn_like(frames)
        noisy_frames = (1 - t.view(-1, 1, 1, 1, 1)) * frames + t.view(-1, 1, 1, 1, 1) * noise
        
        # Process conditions
        text_context = conditions.get('text')
        
        # Model prediction
        pred = self.wan_t2v.model(
            noisy_frames,
            t,
            context=[text_context] if text_context is not None else None,
            seq_len=self._compute_seq_len(noisy_frames.shape),
            extra_conditions=conditions
        )[0]
        
        # In flow matching, target is the original data
        loss = torch.mean((pred - frames) ** 2 * (1 - t.view(-1, 1, 1, 1, 1)))
        logger.debug(f"Flow matching loss: {loss.item()}")
        
        return loss
    

# Example usage
if __name__ == "__main__":
    model = OmniHumanWanT2V(
        wan_config=t2v_14B,
        checkpoint_dir="./checkpoints",
        num_frames=49,
        device_id=0
    )
    text_prompt = "A person dancing in a vibrant city square."
    video = model(
        text_prompt=text_prompt,
        num_inference_steps=50,
        cfg_scale=7.5,
        seed=42
    )
    print(f"Generated video shape: {video.shape}")