import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple, Union
import os
import json
import math
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import json
import cv2
import mediapipe as mp
from tqdm import tqdm

class OmniHumanDataset(Dataset):
    """Dataset for OmniHuman training with mixed condition support and MediaPipe-based heatmap generation."""
    
    def __init__(
        self,
        data_dir: str,
        condition_ratios: Dict[str, float],
        num_frames: int = 48,
        frame_size: Tuple[int, int] = (256, 256),
        num_keypoints: int = 33,  # MediaPipe Pose outputs 33 keypoints
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 2.0,
        transform: Optional[T.Compose] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.condition_ratios = condition_ratios
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Setup transforms for frames
        if transform is None:
            self.transform = T.Compose([
                T.Resize(frame_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # Load dataset annotations
        self.annotations = self._load_annotations()
        
        # Filter data based on quality and condition availability
        self.filtered_data = self._filter_data()
        
    def _load_annotations(self) -> Dict:
        """Load dataset annotations."""
        annotation_file = self.data_dir / 'annotations.json'
        with open(annotation_file, 'r') as f:
            return json.load(f)
            
    def _filter_data(self) -> List[Dict]:
        """Filter data based on quality and available conditions."""
        filtered = []
        for item in self.annotations:
            if item['quality_score'] < 0.7:
                continue
            if 'motion_score' in item and item['motion_score'] < 0.5:
                continue
            valid_conditions = True
            for cond, ratio in self.condition_ratios.items():
                if ratio > 0 and not item.get(f'has_{cond}', False):
                    valid_conditions = False
                    break
            if valid_conditions:
                filtered.append(item)
        return filtered
        
    def _load_frames(self, video_path: Path) -> torch.Tensor:
        """Load and preprocess video frames."""
        frames = []
        frame_files = sorted(list(video_path.glob('*.jpg')))
        
        if len(frame_files) > self.num_frames:
            frame_indices = np.linspace(0, len(frame_files)-1, self.num_frames, dtype=int)
            frame_files = [frame_files[i] for i in frame_indices]
        
        for frame_file in frame_files[:self.num_frames]:
            frame = Image.open(frame_file).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
            
        return torch.stack(frames)  # [T, C, H, W]
        
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and process audio features (placeholder)."""
        return torch.randn(self.num_frames, 1024)  # Mock wav2vec output
        
    def _extract_keypoints(self, frame: torch.Tensor) -> np.ndarray:
        """Extract 2D keypoints from a frame using MediaPipe."""
        # Convert frame tensor [C, H, W] to numpy [H, W, C] for MediaPipe
        frame_np = frame.permute(1, 2, 0).numpy() * 0.5 + 0.5  # Denormalize
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Detect keypoints with MediaPipe
        results = self.mp_pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = np.array([
                [lm.x * self.frame_size[1], lm.y * self.frame_size[0], lm.visibility]
                for lm in results.pose_landmarks.landmark
            ])  # [33, 3] for [x, y, visibility]
        else:
            # If no detection, return zeros with low visibility
            keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            keypoints[:, 2] = 0.0  # Set visibility to 0
        
        return keypoints
        
    def _generate_heatmap(self, keypoints: np.ndarray, size: Tuple[int, int], sigma: float) -> np.ndarray:
        """Generate Gaussian heatmaps from keypoints."""
        h, w = size
        heatmap = np.zeros((self.num_keypoints, h, w), dtype=np.float32)
        
        for k in range(self.num_keypoints):
            if keypoints[k, 2] > 0.1:  # Visibility threshold
                x, y = keypoints[k, :2]
                x = int(x * w / self.frame_size[1])  # Scale to heatmap size
                y = int(y * h / self.frame_size[0])
                
                if 0 <= x < w and 0 <= y < h:
                    grid_y, grid_x = np.ogrid[:h, :w]
                    dist_sq = (grid_x - x)**2 + (grid_y - y)**2
                    heatmap[k] = np.exp(-dist_sq / (2.0 * sigma**2))
        
        return heatmap
    
    def _load_pose(self, frames: torch.Tensor) -> torch.Tensor:
        """Generate pose heatmaps from video frames using MediaPipe."""
        pose_heatmaps = []
        
        for t in range(frames.shape[0]):
            keypoints = self._extract_keypoints(frames[t])  # [33, 3]
            heatmap = self._generate_heatmap(keypoints, self.heatmap_size, self.sigma)
            pose_heatmaps.append(torch.from_numpy(heatmap))
        
        return torch.stack(pose_heatmaps)  # [T, 33, H, W]
        
    def __len__(self) -> int:
        return len(self.filtered_data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.filtered_data[idx]
        data_path = self.data_dir / item['relative_path']
        
        # Load frames
        frames = self._load_frames(data_path / 'frames')
        
        # Load conditions based on availability
        conditions = {}
        
        if item.get('has_audio', False):
            conditions['audio'] = self._load_audio(data_path / 'audio.wav')
            
        if item.get('has_pose', False):
            conditions['pose'] = self._load_pose(frames)  # Generate heatmaps with MediaPipe
            
        if item.get('has_text', False):
            conditions['text'] = torch.tensor(item['text_embedding'])
            
        if item.get('has_reference', False):
            ref_path = data_path / 'reference.jpg'
            reference = Image.open(ref_path).convert('RGB')
            conditions['reference'] = self.transform(reference)
            
        return {
            'frames': frames,
            'conditions': conditions,
            'metadata': {
                'id': item['id'],
                'duration': item['duration'],
                'fps': item['fps']
            }
        }

    def __del__(self):
        """Clean up MediaPipe resources."""
        self.mp_pose.close()

import wandb        
class OmniHumanTrainer:
    """Handles multi-stage training process for OmniHuman with flow matching."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cuda",
        output_dir: str = "outputs",
        local_rank: int = 0
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.local_rank = local_rank
        
        self.setup_distributed()
        self.setup_optimizers()
        self.setup_logging()
        
    def setup_distributed(self):
        """Setup distributed training."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            
    def setup_optimizers(self):
        """Initialize optimizers and schedulers."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['total_steps']
        )
        
    def setup_logging(self):
        """Setup training logging with file output and optional W&B integration."""
        if self.local_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Setup file logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(message)s',
                handlers=[
                    logging.FileHandler(self.output_dir / 'training.log'),
                    logging.StreamHandler()
                ]
            )
            
            # Initialize W&B if enabled
            if self.use_wandb:
                wandb.init(
                    project="OmniHuman",
                    config=self.config,
                    name=f"run_stage_{self.local_rank}_{self.config.get('run_name', 'default')}",
                    dir=str(self.output_dir),
                    group="training" if self.local_rank == 0 else None,  # Group runs in distributed setup
                    reinit=True  # Allow multiple runs in the same process
                )
                wandb.watch(self.model, log="all", log_freq=100)  # Log gradients and parameters
            
    def flow_matching_loss(self, x: torch.Tensor, pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss (simplified version of ODE-based objective)."""
        # Flow matching aims to match the velocity field; here we use a simple MSE
        # between predicted noise and target noise, adjusted by time
        target = x  # In flow matching, target is the data itself at t=0
        loss = torch.mean((pred - target) ** 2 * (1 - t.view(-1, 1, 1, 1, 1)))
        return loss
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise using a flow matching schedule."""
        batch_size = x.shape[0]
        noise = torch.randn_like(x, device=self.device)
        # Linear interpolation for simplicity (could use cosine schedule)
        noisy_x = (1 - t.view(-1, 1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1, 1) * noise
        return noisy_x, noise
    
    def training_step(
        self,
        frames: torch.Tensor,
        conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute single training step with flow matching."""
        batch_size = frames.shape[0]
        t = torch.rand(batch_size, device=self.device)  # t in [0, 1]
        
        # Add noise
        noisy_frames, noise = self.add_noise(frames, t)
        
        # Model prediction
        pred = self.model(
            noisy_frames,
            t,  # Timesteps as continuous values
            **conditions
        )
        
        # Compute flow matching loss
        loss = self.flow_matching_loss(frames, pred, t)
        return loss
        
    def train_stage(
        self,
        stage: int,
        dataloader: DataLoader,
        condition_ratios: Dict[str, float]
    ) -> None:
        """Run a single training stage."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Training Stage {stage}"):
            self.optimizer.zero_grad()
            
            # Move data to device
            frames = batch['frames'].to(self.device)
            conditions = {k: v.to(self.device) for k, v in batch['conditions'].items()}
            
            # Apply condition ratios
            active_conditions = {}
            for cond_type, ratio in condition_ratios.items():
                if ratio > 0 and cond_type in conditions:
                    if torch.rand(1).item() < ratio:
                        active_conditions[cond_type] = conditions[cond_type]
            
            # Forward and backward pass
            loss = self.training_step(frames, active_conditions)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if self.local_rank == 0 and num_batches % self.config['log_interval'] == 0:
                avg_loss = total_loss / num_batches
                logging.info(f"Stage {stage} | Batch {num_batches} | Loss: {avg_loss:.4f}")
                
        if self.local_rank == 0:
            self.save_checkpoint(stage, num_batches)
            
    def save_checkpoint(self, stage: int, step: int):
        """Save model checkpoint."""
        if self.local_rank == 0:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'stage': stage,
                'step': step,
                'config': self.config
            }
            torch.save(
                checkpoint,
                self.output_dir / f'checkpoint_stage{stage}_step{step}.pt'
            )
            
    def train(self, data_config: Dict) -> None:
        """Execute complete training process with all stages per OmniHuman spec."""
        # Stage 1: Text and reference only (weakest conditions)
        stage1_ratios = {
            'text': 1.0,
            'reference': 1.0,
            'audio': 0.0,
            'pose': 0.0
        }
        dataset = OmniHumanDataset(
            data_config['data_dir'],
            stage1_ratios,
            num_frames=self.config['num_frames'],
            num_keypoints=33  # Match MediaPipe
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        self.train_stage(1, dataloader, stage1_ratios)
        
        # Stage 2: Add audio, drop pose
        stage2_ratios = {
            'text': 1.0,
            'reference': 1.0,
            'audio': 0.5,  # Halved from text/reference
            'pose': 0.0
        }
        dataset = OmniHumanDataset(
            data_config['data_dir'],
            stage2_ratios,
            num_frames=self.config['num_frames'],
            num_keypoints=33
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        self.train_stage(2, dataloader, stage2_ratios)
        
        # Stage 3: All conditions with balanced ratios
        stage3_ratios = {
            'text': 1.0,    # Weakest condition, full ratio
            'reference': 1.0,
            'audio': 0.25,  # Stronger, reduced ratio
            'pose': 0.13   # Strongest, lowest ratio
        }
        dataset = OmniHumanDataset(
            data_config['data_dir'],
            stage3_ratios,
            num_frames=self.config['num_frames'],
            num_keypoints=33
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        self.train_stage(3, dataloader, stage3_ratios)

# Example config
if __name__ == "__main__":
    from omni_human import OmniHuman  # Assuming previous model code is in a file

    config = {
        'learning_rate': 1e-4,
        'total_steps': 100000,
        'batch_size': 4,
        'num_frames': 16,
        'num_workers': 4,
        'log_interval': 100
    }
    data_config = {'data_dir': 'path/to/data'}
    
    model = OmniHuman(num_frames=16)
    trainer = OmniHumanTrainer(model, config)
    trainer.train(data_config)