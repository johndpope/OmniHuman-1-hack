from logger import logger
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
from keypoint_processor import SapiensKeypointProcessor
import argparse
from omnihuman_wan_t2v import OmniHumanWanT2V
from omnihuman_dataset import OmniHumanDataset 
from video_tracker import VideoEventData, VideoEvent, ProblematicVideoTracker

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
        self.output_dir = Path(output_dir or config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tracker for problematic videos
        tracker_dir = Path(config.data.get('data_dir', './data')) / "bad_videos"
        tracker_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ProblematicVideoTracker(tracker_dir)
        
        # Save config
        OmegaConf.save(config, self.output_dir / "config.yaml")
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            mixed_precision=config.get('mixed_precision', 'no'),
            log_with="wandb" if config.get('use_wandb', False) else None
        )
        
        self.is_main_process = self.accelerator.is_main_process
        
        # Calculate total steps if not explicitly provided
        if not hasattr(config, 'total_steps'):
            # Sum steps from all stages
            total_steps = sum(stage.get('num_steps', 10000) for stage in config.get('stages', []))
            logger.info(f"Auto-calculated total_steps: {total_steps} from all training stages")
            # Add to config
            config.total_steps = total_steps
        
        # Initialize optimizer and learning rate scheduler
        self.setup_optimizers()
        
        # Setup logging
        self.setup_logging()
        
        # Prepare model, optimizer, and scheduler with accelerator
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        logger.info(f"Initialized OmniHumanTrainer with config: {OmegaConf.to_yaml(config)}")
    
    def setup_optimizers(self):
        """Initialize optimizer and learning rate scheduler."""
        # Get optimizer parameters
        optimizer_type = self.config.get('optimizer_type', 'AdamW')
        optimizer_cls = getattr(torch.optim, optimizer_type)
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999))
        )
        
        # Get scheduler parameters
        total_steps = self.config.get('total_steps', 100000)
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.get('end_factor', 0.1),
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
                eta_min=self.config.get('min_lr', 1e-6)
            )
        
        logger.info(f"Setup optimizer {optimizer_cls.__name__} with scheduler {scheduler_type}")
        
    def setup_logging(self):
        """Setup training logging with wandb integration."""
        if self.is_main_process:
            # Initialize W&B if enabled
            if self.config.get('use_wandb', False):
                run_name = self.config.get('run_name')
                if run_name is None:
                    model_type = self.config.get('model_type', 'default')
                    run_name = f"omnihuman-{model_type}-{wandb.util.generate_id()}"
                
                # Create init kwargs with safe fallbacks
                wandb_kwargs = {
                    "name": run_name,
                    "dir": str(self.output_dir),
                }
                
                # Add optional group and tags if they exist
                if hasattr(self.config, 'wandb_group') and self.config.wandb_group is not None:
                    wandb_kwargs["group"] = self.config.wandb_group
                    
                if hasattr(self.config, 'wandb_tags') and self.config.wandb_tags is not None:
                    wandb_kwargs["tags"] = self.config.wandb_tags
                
                self.accelerator.init_trackers(
                    project_name=self.config.get('wandb_project', 'OmniHuman'),
                    config=OmegaConf.to_container(self.config, resolve=True),
                    init_kwargs={"wandb": wandb_kwargs}
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
    
    def prepare_dataset(self, stage_config: DictConfig):
        """Prepare dataset with the appropriate condition ratios.
        
        Args:
            stage_config: Configuration for the current stage
            
        Returns:
            Configured DataLoader
        """
        # Get condition ratios for this stage
        condition_ratios = stage_config.get('condition_ratios', {})
        logger.info(f"Preparing dataset with condition ratios: {condition_ratios}")
    
        
        dataset = OmniHumanDataset(
                config=self.config,
                condition_ratios=condition_ratios
        )
            
        
        # Ensure we have at least one sample
        if len(dataset) == 0:
            logger.error("Dataset is empty! Trying to use mock data.")
            try:
                from mock_dataset import MockOmniHumanDataset
                dataset = MockOmniHumanDataset(
                    config=self.config, 
                    condition_ratios=condition_ratios,
                    num_samples=self.config.get('mock_samples', 32)
                )
            except ImportError:
                raise ValueError("Dataset is empty.")
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.config.get('batch_size', 4), len(dataset)),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
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
                            self.config.get('max_grad_norm', 1.0)
                        )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                accumulated_loss += self.accelerator.gather(loss).mean().item()
                num_batches += 1
                
                # Log metrics at specified intervals
                if step % self.config.get('log_interval', 100) == 0:
                    avg_loss = accumulated_loss / max(num_batches, 1)
                    metrics = {
                        "loss": avg_loss,
                    }
                    self.log_metrics(metrics, step, stage)
                    
                    # Reset metrics
                    accumulated_loss = 0.0
                    num_batches = 0
                
                # Save checkpoint at specified intervals
                if step % self.config.get('checkpoint_interval', 1000) == 0:
                    self.save_checkpoint(step, stage)
                
                # Update progress bar
                progress_bar.update(1)
                step += 1
                
                # Break if we've reached the desired number of steps
                if step >= num_steps:
                    break
        
        # Save final checkpoint for this stage
        is_final = (stage == len(self.config.get('stages', [])))
        self.save_checkpoint(step, stage, is_final=is_final)
        logger.info(f"Completed training stage {stage}")
    
    def train(self):
        """Execute complete training process with all stages per OmniHuman spec."""
        logger.info("Starting OmniHuman training")
        
        # Get stage configurations
        stages = self.config.get('stages', [])
        num_stages = len(stages)
        
        if num_stages == 0:
            logger.warning("No training stages defined in config")
            return
            
        logger.info(f"Training will proceed in {num_stages} stages")
        
        for stage_idx, stage_config in enumerate(stages):
            stage_num = stage_idx + 1
            logger.info(f"Preparing stage {stage_num}/{num_stages}: {stage_config.get('name', f'stage_{stage_num}')}")
            
            # Get condition ratios and steps for this stage with fallbacks
            condition_ratios = stage_config.get('condition_ratios', {})
            num_steps = stage_config.get('num_steps', 10000)
            
            # Create dataset and dataloader for this stage
            dataloader = self.prepare_dataset(stage_config)
            
            # Train for this stage
            self.train_stage(
                stage=stage_num,
                dataloader=dataloader,
                condition_ratios=condition_ratios,
                num_steps=num_steps
            )
            
        logger.info("Training completed")
        
        # Save problematic videos summary
        self.tracker.save_summary()
        self.tracker.print_summary()
        
        # Final cleanup
        if self.config.get('use_wandb', False) and self.is_main_process:
            wandb.finish()



def parse_args():
    parser = argparse.ArgumentParser(description="Train or run inference with OmniHuman")
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
    config = load_config('omni_config.yaml')
    
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
        trainer.train()
        
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