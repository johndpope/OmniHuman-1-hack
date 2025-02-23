import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import math
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

class APTImageDataset(Dataset):
    """Dataset for image training phase."""
    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        image_size: int = 1024,
        transform: Optional[T.Compose] = None,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Load captions
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
            
        # Get image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                list(self.image_dir.glob('*.png')))
        
        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Get caption
        caption = self.captions[image_path.stem]
        
        return image, caption

class APTVideoDataset(Dataset):
    """Dataset for video training phase."""
    def __init__(
        self,
        video_dir: str,
        caption_file: str,
        num_frames: int = 48,
        frame_width: int = 1280,
        frame_height: int = 720,
        transform: Optional[T.Compose] = None,
    ):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Load captions
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
            
        # Get video directories (each containing frame sequences)
        self.video_dirs = [d for d in self.video_dir.iterdir() if d.is_dir()]
        
        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((frame_height, frame_width), interpolation=T.InterpolationMode.LANCZOS),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        video_dir = self.video_dirs[idx]
        frame_files = sorted(list(video_dir.glob('*.jpg')) + 
                           list(video_dir.glob('*.png')))
        
        # Ensure we have enough frames
        if len(frame_files) < self.num_frames:
            raise ValueError(f"Video {video_dir} has fewer frames than required")
        
        # Randomly select consecutive frames if we have more than needed
        if len(frame_files) > self.num_frames:
            start_idx = np.random.randint(0, len(frame_files) - self.num_frames)
            frame_files = frame_files[start_idx:start_idx + self.num_frames]
        
        # Load and process frames
        frames = []
        for frame_path in frame_files:
            frame = Image.open(frame_path).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
            
        # Stack frames into tensor [T, C, H, W]
        video = torch.stack(frames)
        
        # Get caption
        caption = self.captions[video_dir.name]
        
        return video, caption

# class APTEvaluator:
#     """Handles evaluation metrics for APT model."""
#     def __init__(
#         self,
#         device: str = "cuda",
#         real_stats_file: Optional[str] = None
#     ):
#         self.device = device
        
#         # Initialize metrics
#         self.fid = FrechetInceptionDistance(normalize=True).to(device)
#         self.inception_score = InceptionScore(normalize=True).to(device)
#         self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
        
#         # Load pre-computed real statistics if provided
#         if real_stats_file and os.path.exists(real_stats_file):
#             self.real_stats = torch.load(real_stats_file)
#         else:
#             self.real_stats = None

#     def precompute_real_stats(
#         self,
#         dataloader: DataLoader,
#         save_path: Optional[str] = None
#     ) -> Dict[str, torch.Tensor]:
#         """Pre-compute statistics for real images/videos."""
#         self.fid.reset()
        
#         # Compute stats for real data
#         for batch, _ in dataloader:
#             if len(batch.shape) == 5:  # video data
#                 # Reshape video batch to image batch
#                 batch = batch.view(-1, *batch.shape[2:])
            
#             batch = batch.to(self.device)
#             self.fid.update(batch, real=True)
            
#         # Store real stats
#         self.real_stats = {
#             'mu': self.fid.real_features_sum / self.fid.real_features_num_samples,
#             'sigma': self.fid.real_features_cov_sum / self.fid.real_features_num_samples
#         }
        
#         # Save if path provided
#         if save_path:
#             torch.save(self.real_stats, save_path)
            
#         return self.real_stats

#     @torch.no_grad()
#     def compute_metrics(
#         self,
#         generator: nn.Module,
#         eval_batch: Tuple[torch.Tensor, List[str]],
#         num_samples: int = 1000
#     ) -> Dict[str, float]:
#         """Compute evaluation metrics for generated samples."""
#         self.fid.reset()
#         self.inception_score.reset()
#         self.clip_score.reset()
        
#         real_samples, captions = eval_batch
#         batch_size = real_samples.size(0)
        
#         # Generate samples
#         noise = torch.randn(batch_size, *real_samples.shape[1:], device=self.device)
#         fake_samples = generator(noise, captions)
        
#         # Handle video data
#         if len(fake_samples.shape) == 5:
#             fake_samples = fake_samples.view(-1, *fake_samples.shape[2:])
#             real_samples = real_samples.view(-1, *real_samples.shape[2:])
        
#         # Update metrics
#         self.fid.update(fake_samples, real=False)
#         if self.real_stats is not None:
#             self.fid.real_features_sum = self.real_stats['mu'] * self.fid.real_features_num_samples
#             self.fid.real_features_cov_sum = self.real_stats['sigma'] * self.fid.real_features_num_samples
#         else:
#             self.fid.update(real_samples, real=True)
            
#         self.inception_score.update(fake_samples)
        
#         # Compute CLIP score for text-image alignment
#         clip_scores = []
#         for i in range(batch_size):
#             score = self.clip_score(fake_samples[i:i+1], [captions[i]])
#             clip_scores.append(score)
            
#         # Compute all metrics
#         metrics = {
#             'fid': float(self.fid.compute()),
#             'inception_score': float(self.inception_score.compute()[0]),
#             'clip_score': float(torch.mean(torch.stack(clip_scores)))
#         }
        
#         return metrics

# class APTEvalCallback:
#     """Callback for periodic evaluation during training."""
#     def __init__(
#         self,
#         evaluator: APTEvaluator,
#         eval_dataloader: DataLoader,
#         eval_interval: int = 100,
#         num_samples: int = 1000
#     ):
#         self.evaluator = evaluator
#         self.eval_dataloader = eval_dataloader
#         self.eval_interval = eval_interval
#         self.num_samples = num_samples
#         self.best_metrics = {'fid': float('inf')}
        
#     def __call__(
#         self,
#         trainer: 'APTTrainer',
#         step: int,
#         phase: str
#     ) -> Dict[str, float]:
#         """Run evaluation if at appropriate interval."""
#         if step % self.eval_interval != 0:
#             return {}
            
#         # Get evaluation batch
#         eval_batch = next(iter(self.eval_dataloader))
        
#         # Compute metrics
#         metrics = self.evaluator.compute_metrics(
#             trainer.generator,
#             eval_batch,
#             self.num_samples
#         )
        
#         # Update best metrics
#         if metrics['fid'] < self.best_metrics['fid']:
#             self.best_metrics = metrics.copy()
            
#             # Save best model
#             trainer.save_checkpoint(
#                 step=step,
#                 phase=phase,
#                 suffix='best',
#                 metrics=metrics
#             )
            
#         return metrics

# def setup_data_and_eval(config: 'APTConfig') -> Tuple[DataLoader, DataLoader, APTEvaluator]:
#     """Setup data loaders and evaluator."""
    
#     # Create datasets
#     image_dataset = APTImageDataset(
#         image_dir=config.image_dir,
#         caption_file=config.image_caption_file,
#         image_size=config.image_size
#     )
    
#     video_dataset = APTVideoDataset(
#         video_dir=config.video_dir,
#         caption_file=config.video_caption_file,
#         num_frames=config.video_frames,
#         frame_width=config.video_width,
#         frame_height=config.video_height
#     )
    
#     # Create dataloaders
#     image_dataloader = DataLoader(
#         image_dataset,
#         batch_size=config.image_batch_size // config.world_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     video_dataloader = DataLoader(
#         video_dataset,
#         batch_size=config.video_batch_size // config.world_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Create evaluator
#     evaluator = APTEvaluator(
#         device=config.device,
#         real_stats_file=config.real_stats_file
#     )
    
#     return image_dataloader, video_dataloader, evaluator
