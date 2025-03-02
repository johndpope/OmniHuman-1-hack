import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import random
import wandb
from torchvision.utils import make_grid
import cv2
from einops import rearrange

class VideoAugmentation:
    """Advanced video augmentation techniques."""
    
    def __init__(
        self,
        temporal_crop_prob: float = 0.5,
        temporal_mask_prob: float = 0.3,
        color_jitter_prob: float = 0.3,
        gaussian_blur_prob: float = 0.2,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        blur_kernel_range: Tuple[int, int] = (3, 7)
    ):
        self.temporal_crop_prob = temporal_crop_prob
        self.temporal_mask_prob = temporal_mask_prob
        self.color_jitter_prob = color_jitter_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.blur_kernel_range = blur_kernel_range

    def apply_temporal_crop(
        self,
        video: torch.Tensor,
        min_keep: float = 0.8
    ) -> torch.Tensor:
        """Randomly crop video in temporal dimension."""
        if random.random() > self.temporal_crop_prob:
            return video
            
        T = video.size(0)
        keep_length = int(T * random.uniform(min_keep, 1.0))
        start_idx = random.randint(0, T - keep_length)
        
        # Crop and resize back to original length
        video = video[start_idx:start_idx + keep_length]
        return F.interpolate(
            video.unsqueeze(0),
            size=(T, *video.shape[2:]),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

    def apply_temporal_mask(
        self,
        video: torch.Tensor,
        max_mask_length: int = 4
    ) -> torch.Tensor:
        """Apply random temporal masking."""
        if random.random() > self.temporal_mask_prob:
            return video
            
        T = video.size(0)
        mask_length = random.randint(1, min(max_mask_length, T // 4))
        start_idx = random.randint(0, T - mask_length)
        
        # Create mask
        video_masked = video.clone()
        video_masked[start_idx:start_idx + mask_length] = 0
        return video_masked

    def apply_color_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """Apply consistent color jittering across frames."""
        if random.random() > self.color_jitter_prob:
            return video
            
        # Sample random color transformations
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        saturation = random.uniform(*self.saturation_range)
        hue = random.uniform(*self.hue_range)
        
        # Apply consistently to all frames
        video = TF.adjust_brightness(video, brightness)
        video = TF.adjust_contrast(video, contrast)
        video = TF.adjust_saturation(video, saturation)
        video = TF.adjust_hue(video, hue)
        
        return video

    def apply_gaussian_blur(self, video: torch.Tensor) -> torch.Tensor:
        """Apply temporal-consistent Gaussian blur."""
        if random.random() > self.gaussian_blur_prob:
            return video
            
        # Sample kernel size
        kernel_size = random.randrange(
            self.blur_kernel_range[0],
            self.blur_kernel_range[1],
            2
        )
        
        sigma = random.uniform(0.1, 2.0)
        
        # Apply to all frames
        frames = []
        for frame in video:
            frame = TF.gaussian_blur(
                frame.unsqueeze(0),
                kernel_size,
                sigma
            )
            frames.append(frame)
            
        return torch.cat(frames)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations in sequence."""
        video = self.apply_temporal_crop(video)
        video = self.apply_temporal_mask(video)
        video = self.apply_color_jitter(video)
        video = self.apply_gaussian_blur(video)
        return video

class TemporalConsistencyLoss(nn.Module):
    """Loss function for temporal consistency in videos."""
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        if len(video.shape) == 5:  # [B, T, C, H, W]
            video = rearrange(video, 'b t c h w -> b c t h w')
            
        # Compute frame differences
        frame_diffs = video[..., 1:, :, :] - video[..., :-1, :, :]
        
        # Compute flow-based consistency
        consistency_loss = torch.mean(torch.abs(frame_diffs))
        
        return self.weight * consistency_loss

class APTTrainingMonitor:
    """Monitors and visualizes APT training progress."""
    
    def __init__(
        self,
        config: 'APTConfig',
        log_dir: str = "logs",
        use_wandb: bool = True
    ):
        self.config = config
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        if self.use_wandb and config.local_rank == 0:
            wandb.init(
                project="apt-training",
                config=vars(config),
                dir=log_dir
            )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str
    ):
        """Log training metrics."""
        if self.config.local_rank != 0:
            return
            
        # Prepare metrics for logging
        log_dict = {f"{phase}/{k}": v for k, v in metrics.items()}
        
        if self.use_wandb:
            wandb.log(log_dict, step=step)

    def log_samples(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        step: int,
        phase: str,
        num_samples: int = 4
    ):
        """Log sample images/videos for visualization."""
        if self.config.local_rank != 0:
            return
            
        # Handle both image and video batches
        is_video = len(real_batch.shape) == 5
        
        if is_video:
            # Sample frames for visualization
            real_frames = self._sample_video_frames(real_batch, num_samples)
            fake_frames = self._sample_video_frames(fake_batch, num_samples)
            
            # Create video grid
            video_grid = self._create_video_grid(
                real_frames,
                fake_frames,
                nrow=num_samples
            )
            
            if self.use_wandb:
                wandb.log({
                    f"{phase}/video_samples": wandb.Video(
                        video_grid.permute(1, 0, 2, 3).cpu().numpy(),
                        fps=4
                    )
                }, step=step)
        else:
            # Create image grid
            samples = torch.cat([
                real_batch[:num_samples],
                fake_batch[:num_samples]
            ])
            grid = make_grid(samples, nrow=num_samples, normalize=True)
            
            if self.use_wandb:
                wandb.log({
                    f"{phase}/image_samples": wandb.Image(grid)
                }, step=step)

    def _sample_video_frames(
        self,
        batch: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """Sample frames from video batch for visualization."""
        B, T, C, H, W = batch.shape
        num_samples = min(num_samples, B)
        
        # Sample evenly spaced frames
        frame_indices = torch.linspace(0, T-1, 8).long()
        return batch[:num_samples, frame_indices]

    def _create_video_grid(
        self,
        real_frames: torch.Tensor,
        fake_frames: torch.Tensor,
        nrow: int
    ) -> torch.Tensor:
        """Create a grid of video frames for visualization."""
        B, T, C, H, W = real_frames.shape
        
        # Combine real and fake
        all_frames = torch.cat([real_frames, fake_frames])
        
        # Create grid for each timestep
        grid_frames = []
        for t in range(T):
            grid = make_grid(
                all_frames[:, t],
                nrow=nrow,
                normalize=True
            )
            grid_frames.append(grid)
            
        return torch.stack(grid_frames)

class APTVideoProcessor:
    """Handles video processing and frame interpolation."""
    
    def __init__(
        self,
        target_fps: int = 24,
        interpolation_factor: int = 2
    ):
        self.target_fps = target_fps
        self.interpolation_factor = interpolation_factor

    def load_video(self, path: str) -> torch.Tensor:
        """Load video and convert to target FPS."""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        
        # Convert to tensor
        video = torch.from_numpy(np.stack(frames))
        
        # Adjust FPS if needed
        if fps != self.target_fps:
            video = self.adjust_fps(video, fps)
            
        return video

    def adjust_fps(
        self,
        video: torch.Tensor,
        source_fps: float
    ) -> torch.Tensor:
        """Adjust video FPS using interpolation."""
        if source_fps == self.target_fps:
            return video
            
        # Calculate target length
        T = video.size(0)
        target_length = int(T * self.target_fps / source_fps)
        
        # Interpolate
        return F.interpolate(
            video.permute(3, 0, 1, 2).unsqueeze(0),
            size=(target_length, *video.shape[1:3]),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 3, 0)

    def interpolate_frames(
        self,
        video: torch.Tensor
    ) -> torch.Tensor:
        """Perform frame interpolation for smoother video."""
        B, T, C, H, W = video.shape
        target_length = T * self.interpolation_factor
        
        return F.interpolate(
            video.transpose(1, 2),
            size=(target_length, H, W),
            mode='trilinear',
            align_corners=False
        ).transpose(1, 2)

def setup_video_training(
    config: 'APTConfig',
    training_loop: 'APTTrainingLoop'
) -> 'APTTrainingLoop':
    """Setup video-specific training components."""
    
    # Add video augmentation
    video_aug = VideoAugmentation(
        temporal_crop_prob=0.5,
        temporal_mask_prob=0.3
    )
    
    # Add temporal consistency loss
    temporal_loss = TemporalConsistencyLoss(weight=0.5)
    
    # Add training monitor
    monitor = APTTrainingMonitor(config)
    
    # Update training loop
    training_loop.video_aug = video_aug
    training_loop.temporal_loss = temporal_loss
    training_loop.monitor = monitor
    
    return training_loop
