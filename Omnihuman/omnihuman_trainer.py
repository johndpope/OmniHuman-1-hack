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
import argparse
from omnihuman_wan_t2v import OmniHumanWanT2V
from omnihuman_dataset import OmniHumanDataset 

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
    
    

class OmniHumanTrainer:
    """Enhanced training manager for OmniHuman with accelerate, wandb, and omegaconf support."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        output_dir: Optional[str] = None,
    ):
        """Initialize the OmniHuman trainer.
        
        Args:
            model: The OmniHuman model
            config: Training configuration
            output_dir: Output directory for logs and checkpoints
        """
        self.model = model
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(output_dir or config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        OmegaConf.save(config, self.output_dir / "config.yaml")
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            mixed_precision=config.get("mixed_precision", "no"),
            log_with="wandb" if config.get("use_wandb", False) else None
        )
        
        self.is_main_process = self.accelerator.is_main_process
        
        # Initialize optimizer and learning rate scheduler
        self.setup_optimizers()
        
        # Setup logging
        self.setup_logging()
        
        # Prepare model, optimizer, and scheduler with accelerator
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        
        # Set random seed for reproducibility
        set_seed(config.get("seed", 42))
        
        logger.info(f"Initialized OmniHumanTrainer with config: {OmegaConf.to_yaml(config)}")
        
    def setup_optimizers(self):
        """Initialize optimizer and learning rate scheduler."""
        # Get optimizer parameters
        optimizer_cls = getattr(torch.optim, self.config.get("optimizer_type", "AdamW"))
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.999))
        )
        
        # Get scheduler parameters
        total_steps = self.config.get("total_steps", 100000)
        scheduler_type = self.config.get("scheduler_type", "cosine")
        
        if scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.get("min_lr", 1e-6)
            )
        elif scheduler_type == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.get("end_factor", 0.1),
                total_iters=total_steps
            )
        elif scheduler_type == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_steps
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using cosine")
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.get("min_lr", 1e-6)
            )
        
        logger.info(f"Setup optimizer {optimizer_cls.__name__} with scheduler {scheduler_type}")
        
    def setup_logging(self):
        """Setup training logging with wandb integration."""
        if self.is_main_process:
           
            # Initialize W&B if enabled
            if self.config.get("use_wandb", False):
                run_name = self.config.get("run_name", None)
                if run_name is None:
                    run_name = f"omnihuman-{self.config.get('model_type', 'default')}-{wandb.util.generate_id()}"
                
                self.accelerator.init_trackers(
                    project_name=self.config.get("wandb_project", "OmniHuman"),
                    config=OmegaConf.to_container(self.config, resolve=True),
                    init_kwargs={
                        "wandb": {
                            "name": run_name,
                            "dir": str(self.output_dir),
                            "group": self.config.get("wandb_group", None),
                            "tags": self.config.get("wandb_tags", []),
                        }
                    }
                )
                logger.info(f"Initialized wandb with run name: {run_name}")
    
    def save_checkpoint(self, step: int, stage: int = None, is_final: bool = False):
        """Save model checkpoint.
        
        Args:
            step: Current training step
            stage: Training stage number (optional)
            is_final: Whether this is the final checkpoint
        """
        if not self.is_main_process:
            return
            
        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'step': step,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        if stage is not None:
            checkpoint['stage'] = stage
            
        # Create checkpoint filename
        if is_final:
            checkpoint_path = self.output_dir / "model_final.pt"
        elif stage is not None:
            checkpoint_path = self.output_dir / f"checkpoint_stage{stage}_step{step}.pt"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_step{step}.pt"
            
        # Save checkpoint
        self.accelerator.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint separately
        latest_path = self.output_dir / "model_latest.pt"
        self.accelerator.save(checkpoint, latest_path)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, stage: Optional[int] = None):
        """Log metrics to wandb and console.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            stage: Current training stage (optional)
        """
        if not self.is_main_process:
            return
            
        # Add prefix to metrics if stage is provided
        if stage is not None:
            metrics = {f"stage_{stage}/{k}": v for k, v in metrics.items()}
            
        # Add learning rate
        metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        
        # Log to accelerator (which handles wandb)
        self.accelerator.log(metrics, step=step)
        
        # Log to console
        log_str = f"Step {step}"
        if stage is not None:
            log_str += f" (Stage {stage})"
        log_str += " | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        logger.info(log_str)
    
    def prepare_dataset(self, stage_config: DictConfig, dataset_cls: type):
        """Prepare dataset with the appropriate condition ratios.
        
        Args:
            stage_config: Configuration for the current stage
            dataset_cls: Dataset class to use
            
        Returns:
            Configured DataLoader
        """
        # Get condition ratios for this stage
        condition_ratios = stage_config.get("condition_ratios", {})
        logger.info(f"Preparing dataset with condition ratios: {condition_ratios}")
        
        # Create dataset
        dataset = dataset_cls(
            data_dir=self.config.data.get("data_dir"),
            condition_ratios=condition_ratios,
            num_frames=self.config.get("num_frames", 49),
            num_keypoints=self.config.get("num_keypoints", 308),  # Sapiens 308 keypoints
            heatmap_size=self.config.data.get("heatmap_size", (64, 64)),
            frame_size=self.config.data.get("frame_size", (256, 256))
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )
        
        return self.accelerator.prepare(dataloader)
    
    def train_stage(
        self,
        stage: int,
        dataloader: DataLoader,
        condition_ratios: Dict[str, float],
        num_steps: int
    ):
        """Train the model for a single stage.
        
        Args:
            stage: Stage number
            dataloader: DataLoader for this stage
            condition_ratios: Condition ratios for this stage
            num_steps: Number of steps to train for
        """
        logger.info(f"Starting training stage {stage} for {num_steps} steps")
        logger.info(f"Condition ratios: {condition_ratios}")
        
        self.model.train()
        step = 0
        
        # Prepare progress bar
        progress_bar = tqdm(
            range(num_steps), 
            disable=not self.is_main_process,
            desc=f"Stage {stage}"
        )
        
        # Initialize metrics
        accumulated_loss = 0.0
        num_batches = 0
        
        # Training loop
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break
                    
                # Get batch data
                frames = batch["frames"]
                conditions = batch["conditions"]
                
                # Apply condition ratios (randomly drop conditions based on ratios)
                active_conditions = {}
                for cond_type, ratio in condition_ratios.items():
                    if ratio > 0 and cond_type in conditions:
                        if torch.rand(1).item() < ratio:
                            active_conditions[cond_type] = conditions[cond_type]
                
                # Generate random timesteps
                batch_size = frames.shape[0]
                t = torch.rand(batch_size, device=frames.device)
                
                # Forward pass with gradient accumulation
                with self.accelerator.accumulate(self.model):
                    # Compute loss
                    loss = self.model.training_step(frames, active_conditions, t)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Optimizer step
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.get("max_grad_norm", 1.0)
                        )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                accumulated_loss += self.accelerator.gather(loss).mean().item()
                num_batches += 1
                
                # Log metrics at specified intervals
                if step % self.config.get("log_interval", 100) == 0:
                    avg_loss = accumulated_loss / max(num_batches, 1)
                    metrics = {
                        "loss": avg_loss,
                    }
                    self.log_metrics(metrics, step, stage)
                    
                    # Reset metrics
                    accumulated_loss = 0.0
                    num_batches = 0
                
                # Save checkpoint at specified intervals
                if step % self.config.get("checkpoint_interval", 1000) == 0:
                    self.save_checkpoint(step, stage)
                
                # Update progress bar
                progress_bar.update(1)
                step += 1
                
                # Break if we've reached the desired number of steps
                if step >= num_steps:
                    break
        
        # Save final checkpoint for this stage
        self.save_checkpoint(step, stage, is_final=(stage == self.config.get("num_stages", 3)))
        logger.info(f"Completed training stage {stage}")
    
    def train(self, dataset_cls: type):
        """Execute complete training process with all stages per OmniHuman spec.
        
        Args:
            dataset_cls: Dataset class to use
        """
        logger.info("Starting OmniHuman training")
        
        # Get stage configurations
        stages = self.config.get("stages", [])
        num_stages = len(stages)
        
        if num_stages == 0:
            logger.warning("No training stages defined in config")
            return
            
        logger.info(f"Training will proceed in {num_stages} stages")
        
        for stage_idx, stage_config in enumerate(stages):
            stage_num = stage_idx + 1
            logger.info(f"Preparing stage {stage_num}/{num_stages}")
            
            # Get condition ratios for this stage
            condition_ratios = stage_config.get("condition_ratios", {})
            num_steps = stage_config.get("num_steps", 10000)
            
            # Create dataset and dataloader for this stage
            dataloader = self.prepare_dataset(stage_config, dataset_cls)
            
            # Train for this stage
            self.train_stage(
                stage=stage_num,
                dataloader=dataloader,
                condition_ratios=condition_ratios,
                num_steps=num_steps
            )
            
        logger.info("Training completed")
        
        # Final cleanup
        if self.config.get("use_wandb", False) and self.is_main_process:
            wandb.finish()



def parse_args():
    parser = argparse.ArgumentParser(description="Train or run inference with OmniHuman")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Operation mode: train or inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="CUDA device ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    config = OmegaConf.load(config_path)
    return config

def load_checkpoint(model: OmniHumanWanT2V, checkpoint_path: str):
    """Load checkpoint into model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("step", 0), checkpoint.get("stage", 0)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug = True
        
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the model
    logger.info("Creating OmniHuman model")
    model = OmniHumanWanT2V(
        config=config,
        device_id=args.device_id,
    )
    
    # Load checkpoint if provided
    start_step = 0
    start_stage = 0
    if args.checkpoint:
        start_step, start_stage = load_checkpoint(model, args.checkpoint)
        
    if args.mode == "train":
        # Create trainer
        logger.info("Creating trainer")
        trainer = OmniHumanTrainer(
            model=model,
            config=config,
            output_dir=output_dir,
        )
        
        # Start training
        logger.info("Starting training")
        trainer.train(OmniHumanDataset)
        
    elif args.mode == "inference":
        if not args.checkpoint:
            logger.warning("No checkpoint provided for inference.")
            
        # Run inference
        logger.info("Running inference")
        
        # Example: Generate a video using audio and pose
        # This is just a placeholder example - you would need to load your actual data
        audio = torch.randn(1, config.num_frames, config.audio_dim).to(model.device)
        pose = torch.randn(1, config.num_frames, config.num_keypoints, 64, 64).to(model.device)
        reference_image = torch.randn(1, 3, 256, 256).to(model.device)
        
        video = model(
            text_prompt="A person talking with natural gestures",
            audio=audio,
            pose=pose,
            reference_image=reference_image,
            num_inference_steps=50,
            cfg_scale=7.5,
            seed=config.seed,
        )
        
        # Save generated video
        # This is a placeholder - you would need to implement video saving
        logger.info(f"Generated video shape: {video.shape}")
        
if __name__ == "__main__":
    main()