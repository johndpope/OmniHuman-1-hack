
import os
import json
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from PIL import Image
import cv2
import librosa
import mediapipe as mp
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import wandb
from logger import logger
from keypoint_processor import SapiensKeypointProcessor


class OmniHumanDataset(Dataset):
    """Dataset for OmniHuman training with Sapiens 308 keypoints and mixed condition support."""
    
    def __init__(
        self,
        data_dir: str,
        condition_ratios: Dict[str, float],
        num_frames: int = 49,
        frame_size: Tuple[int, int] = (256, 256),
        num_keypoints: int = 308,  # Sapiens 308 keypoints
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 2.0,
        transform: Optional[T.Compose] = None,
        audio_sampling_rate: int = 16000,
        audio_features_dim: int = 1024,
        min_quality_score: float = 0.7,
        min_motion_score: float = 0.5,
        sapiens_checkpoints_dir: Optional[str] = None,
        sapiens_model_name: str = "1b",
        sapiens_detection_config: Optional[str] = None,
        sapiens_detection_checkpoint: Optional[str] = None,
    ):
        """Initialize the OmniHuman dataset.
        
        Args:
            data_dir: Directory containing the dataset
            condition_ratios: Ratios for different condition modalities
            num_frames: Number of frames to sample
            frame_size: Size of output frames
            num_keypoints: Number of pose keypoints (308 for Sapiens)
            heatmap_size: Size of pose heatmaps
            sigma: Sigma for Gaussian heatmaps
            transform: Optional transform to apply to frames
            audio_sampling_rate: Audio sampling rate
            audio_features_dim: Dimensionality of audio features
            min_quality_score: Minimum quality score for data filtering
            min_motion_score: Minimum motion score for data filtering
            sapiens_checkpoints_dir: Directory containing Sapiens checkpoints
            sapiens_model_name: Sapiens model size to use
            sapiens_detection_config: Path to detection config file
            sapiens_detection_checkpoint: Path to detection checkpoint file
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.condition_ratios = condition_ratios
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.audio_sampling_rate = audio_sampling_rate
        self.audio_features_dim = audio_features_dim
        self.min_quality_score = min_quality_score
        self.min_motion_score = min_motion_score
        
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
            
        # Setup transform for reference image (no random flip to maintain consistency)
        self.reference_transform = T.Compose([
            T.Resize(frame_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Initialize Sapiens keypoint processor if checkpoints directory is provided
        self.keypoint_processor = None
        if sapiens_checkpoints_dir:
            try:
                self.keypoint_processor = SapiensKeypointProcessor(
                    checkpoints_dir=sapiens_checkpoints_dir,
                    model_name=sapiens_model_name,
                    detection_config=sapiens_detection_config,
                    detection_checkpoint=sapiens_detection_checkpoint,
                    heatmap_size=heatmap_size,
                )
                logger.info(f"Initialized Sapiens keypoint processor with model {sapiens_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Sapiens keypoint processor: {e}")
        else:
            logger.warning("No Sapiens checkpoints directory provided, pose extraction will not be available")
            
        # Load dataset annotations
        self.annotations = self._load_annotations()
        
        # Filter data based on quality and condition availability
        self.filtered_data = self._filter_data()
        
        logger.info(f"Loaded dataset with {len(self.filtered_data)} samples")
        
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations."""
        annotation_file = self.data_dir / 'annotations.json'
        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return []
            
        with open(annotation_file, 'r') as f:
            return json.load(f)
            
    def _filter_data(self) -> List[Dict]:
        """Filter data based on quality and available conditions."""
        filtered = []
        
        for item in tqdm(self.annotations, desc="Filtering data"):
            # Skip if quality is too low
            if item.get('quality_score', 0) < self.min_quality_score:
                continue
                
            # Skip if motion is too low
            if item.get('motion_score', 0) < self.min_motion_score:
                continue
                
            # Check if required conditions are available
            valid_conditions = True
            for cond, ratio in self.condition_ratios.items():
                if ratio > 0 and not item.get(f'has_{cond}', False):
                    valid_conditions = False
                    break
                    
            if valid_conditions:
                filtered.append(item)
                
        logger.info(f"Filtered {len(self.annotations)} samples to {len(filtered)} based on quality and conditions")
        return filtered
        
    def _load_frames(self, video_path: Path) -> torch.Tensor:
        """Load and preprocess video frames."""
        frame_files = sorted(list(video_path.glob('*.jpg')))
        
        if len(frame_files) == 0:
            logger.warning(f"No frames found in {video_path}")
            # Return zeros as fallback
            return torch.zeros((self.num_frames, 3, *self.frame_size))
        
        if len(frame_files) < self.num_frames:
            # If we have fewer frames than needed, repeat the last frame
            frame_files = frame_files + [frame_files[-1]] * (self.num_frames - len(frame_files))
        
        # Sample frames if we have more than needed
        if len(frame_files) > self.num_frames:
            frame_indices = np.linspace(0, len(frame_files)-1, self.num_frames, dtype=int)
            frame_files = [frame_files[i] for i in frame_indices]
        
        frames = []
        for frame_file in frame_files[:self.num_frames]:
            frame = Image.open(frame_file).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
            
        return torch.stack(frames)  # [T, C, H, W]
        
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and process audio features."""
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return torch.zeros((self.num_frames, self.audio_features_dim))
            
        try:
            # In a real implementation, you would use a pretrained wav2vec model here
            # For this example, we'll just return random features
            return torch.randn(self.num_frames, self.audio_features_dim)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return torch.zeros((self.num_frames, self.audio_features_dim))
            
    def _load_text_embedding(self, text_embedding_path: Path) -> torch.Tensor:
        """Load text embedding."""
        if not text_embedding_path.exists():
            logger.warning(f"Text embedding file not found: {text_embedding_path}")
            return torch.zeros(1, 768)  # Default T5 dimension
            
        try:
            embedding = torch.load(text_embedding_path)
            return embedding
        except Exception as e:
            logger.error(f"Error loading text embedding {text_embedding_path}: {e}")
            return torch.zeros(1, 768)
    
    def _extract_keypoints(self, frame_files: List[Path]) -> List[np.ndarray]:
        """Extract Sapiens 308 keypoints from frame files."""
        keypoints_list = []
        
        if not self.keypoint_processor:
            # If no keypoint processor, return empty keypoints
            for _ in range(len(frame_files)):
                keypoints_list.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))
            return keypoints_list
            
        # Process each frame with Sapiens model
        for frame_file in tqdm(frame_files, desc="Extracting keypoints", leave=False):
            try:
                keypoints = self.keypoint_processor.extract_keypoints(frame_file)
                keypoints_list.append(keypoints)
            except Exception as e:
                logger.error(f"Error extracting keypoints from {frame_file}: {e}")
                keypoints_list.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))
                
        return keypoints_list
    
    def _generate_heatmaps(self, keypoints_list: List[np.ndarray]) -> torch.Tensor:
        """Generate heatmaps from keypoints.
        
        Args:
            keypoints_list: List of keypoint arrays [T, K, 3]
            
        Returns:
            Heatmaps tensor [T, K, H, W]
        """
        T = len(keypoints_list)
        K = self.num_keypoints
        H, W = self.heatmap_size
        heatmaps = torch.zeros((T, K, H, W), dtype=torch.float32)
        
        # Either use the keypoint processor's heatmap generator or generate our own
        if self.keypoint_processor:
            for t, keypoints in enumerate(keypoints_list):
                heatmap = self.keypoint_processor.generate_heatmaps(keypoints)
                heatmaps[t] = torch.from_numpy(heatmap)
        else:
            # Manual heatmap generation
            for t, keypoints in enumerate(keypoints_list):
                for k in range(K):
                    if keypoints[k, 2] > 0.1:  # Confidence threshold
                        x, y = keypoints[k, :2]
                        x_scaled = int(x * W)
                        y_scaled = int(y * H)
                        
                        if 0 <= x_scaled < W and 0 <= y_scaled < H:
                            # Create 2D Gaussian
                            grid_y = torch.arange(H, dtype=torch.float32)
                            grid_x = torch.arange(W, dtype=torch.float32)
                            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                            
                            dist_squared = (grid_x - x_scaled)**2 + (grid_y - y_scaled)**2
                            exponent = -dist_squared / (2 * self.sigma**2)
                            heatmap = torch.exp(exponent)
                            heatmaps[t, k] = heatmap
                            
        return heatmaps
    
    def _load_keypoints_from_cache(self, keypoints_path: Path) -> List[np.ndarray]:
        """Load keypoints from cached files."""
        keypoints_list = []
        
        keypoint_files = sorted(list(keypoints_path.glob('*.npy')))
        
        if len(keypoint_files) == 0:
            logger.warning(f"No keypoint files found in {keypoints_path}")
            return [np.zeros((self.num_keypoints, 3), dtype=np.float32) for _ in range(self.num_frames)]
            
        # Sample keypoint files if we have more or fewer than needed
        if len(keypoint_files) < self.num_frames:
            keypoint_files = keypoint_files + [keypoint_files[-1]] * (self.num_frames - len(keypoint_files))
        
        if len(keypoint_files) > self.num_frames:
            indices = np.linspace(0, len(keypoint_files)-1, self.num_frames, dtype=int)
            keypoint_files = [keypoint_files[i] for i in indices]
            
        # Load keypoints from files
        for keypoint_file in keypoint_files:
            try:
                keypoints = np.load(keypoint_file)
                # Ensure we have the right shape
                if keypoints.shape[0] != self.num_keypoints:
                    logger.warning(f"Keypoints in {keypoint_file} have wrong shape: {keypoints.shape}")
                    keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            except Exception as e:
                logger.error(f"Error loading keypoints from {keypoint_file}: {e}")
                keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                
            keypoints_list.append(keypoints)
            
        return keypoints_list
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filtered_data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing frames and conditions
        """
        item = self.filtered_data[idx]
        data_path = self.data_dir / item['relative_path']
        
        # Load frames
        frames_path = data_path / 'frames'
        frames = self._load_frames(frames_path)
        
        # Load conditions based on availability
        conditions = {}
        
        # Always load reference image (first frame)
        reference = frames[0].unsqueeze(0)  # Use first frame as reference
        
        # Load audio if available
        if item.get('has_audio', False) and self.condition_ratios.get('audio', 0) > 0:
            audio_path = data_path / 'audio.wav'
            conditions['audio'] = self._load_audio(audio_path)
            
        # Load pose if available
        if item.get('has_pose', False) and self.condition_ratios.get('pose', 0) > 0:
            # First check if we have cached keypoints
            keypoints_path = data_path / 'keypoints'
            
            if keypoints_path.exists() and list(keypoints_path.glob('*.npy')):
                # Load from cache
                keypoints_list = self._load_keypoints_from_cache(keypoints_path)
            else:
                # Extract keypoints from frames
                frame_files = sorted(list(frames_path.glob('*.jpg')))
                
                if len(frame_files) < self.num_frames:
                    frame_files = frame_files + [frame_files[-1]] * (self.num_frames - len(frame_files))
                
                if len(frame_files) > self.num_frames:
                    indices = np.linspace(0, len(frame_files)-1, self.num_frames, dtype=int)
                    frame_files = [frame_files[i] for i in indices]
                    
                keypoints_list = self._extract_keypoints(frame_files)
            
            # Generate heatmaps from keypoints
            pose_heatmaps = self._generate_heatmaps(keypoints_list)
            conditions['pose'] = pose_heatmaps
            
        # Load text if available
        if item.get('has_text', False) and self.condition_ratios.get('text', 0) > 0:
            text_embedding_path = data_path / 'text_embedding.pt'
            conditions['text'] = self._load_text_embedding(text_embedding_path)
            
        # Load reference if available (otherwise use first frame)
        if item.get('has_reference', False):
            reference_path = data_path / 'reference.jpg'
            if reference_path.exists():
                reference_img = Image.open(reference_path).convert('RGB')
                reference = self.reference_transform(reference_img).unsqueeze(0)
                
        conditions['reference'] = reference
            
        return {
            'frames': frames.permute(1, 0, 2, 3),  # [C, T, H, W] format for 3D models
            'conditions': conditions,
            'metadata': {
                'id': item.get('id', str(idx)),
                'duration': item.get('duration', self.num_frames / 30.0),
                'fps': item.get('fps', 30.0)
            }
        }
    
    