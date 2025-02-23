import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

class APTConfig:
    """Configuration for APT training."""
    def __init__(
        self,
        # Training modes
        train_images: bool = True,
        train_videos: bool = False,
        
        # Data dimensions
        image_size: int = 1024,
        video_width: int = 1280,
        video_height: int = 720,
        video_frames: int = 48,  # 2 seconds at 24fps
        
        # Training hyperparameters
        image_batch_size: int = 9062,
        video_batch_size: int = 2048,
        image_learning_rate: float = 5e-6,
        video_learning_rate: float = 3e-6,
        ema_decay: float = 0.995,
        
        # R1 regularization
        image_r1_sigma: float = 0.01,
        video_r1_sigma: float = 0.1,
        r1_lambda: float = 100.0,
        
        # Training duration
        image_updates: int = 350,
        video_updates: int = 300,
        
        # Distributed training
        world_size: int = 1,
        local_rank: int = 0,
        
        # Hardware
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        
        # Logging and checkpointing
        log_interval: int = 10,
        save_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
    ):
        self.train_images = train_images
        self.train_videos = train_videos
        
        self.image_size = image_size
        self.video_width = video_width
        self.video_height = video_height
        self.video_frames = video_frames
        
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size
        self.image_learning_rate = image_learning_rate
        self.video_learning_rate = video_learning_rate
        self.ema_decay = ema_decay
        
        self.image_r1_sigma = image_r1_sigma
        self.video_r1_sigma = video_r1_sigma
        self.r1_lambda = r1_lambda
        
        self.image_updates = image_updates
        self.video_updates = video_updates
        
        self.world_size = world_size
        self.local_rank = local_rank
        
        self.device = device
        self.dtype = dtype
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        if self.local_rank == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

class APTLogger:
    """Logging utility for APT training."""
    def __init__(self, config: APTConfig):
        self.config = config
        self.reset_metrics()
        
        if config.local_rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s'
            )
            
    def reset_metrics(self):
        """Reset training metrics."""
        self.metrics = {
            "g_loss": [],
            "d_loss": [],
            "r1_loss": [],
        }
        
    def log_metrics(
        self,
        g_loss: float,
        d_loss: float,
        r1_loss: float,
        step: int,
        phase: str
    ):
        """Log training metrics."""
        self.metrics["g_loss"].append(g_loss)
        self.metrics["d_loss"].append(d_loss)
        self.metrics["r1_loss"].append(r1_loss)
        
        if self.config.local_rank == 0 and step % self.config.log_interval == 0:
            avg_g_loss = np.mean(self.metrics["g_loss"][-self.config.log_interval:])
            avg_d_loss = np.mean(self.metrics["d_loss"][-self.config.log_interval:])
            avg_r1_loss = np.mean(self.metrics["r1_loss"][-self.config.log_interval:])
            
            logging.info(
                f"{phase} Step {step}: "
                f"G_loss = {avg_g_loss:.4f}, "
                f"D_loss = {avg_d_loss:.4f}, "
                f"R1_loss = {avg_r1_loss:.4f}"
            )

class APTCheckpointer:
    """Handles model checkpointing."""
    def __init__(self, config: APTConfig):
        self.config = config
        
    def save_checkpoint(
        self,
        trainer: 'APTTrainer',
        step: int,
        phase: str,
        is_final: bool = False
    ):
        """Save model checkpoint."""
        if self.config.local_rank == 0:
            checkpoint = {
                'generator_state_dict': trainer.generator.state_dict(),
                'discriminator_state_dict': trainer.discriminator.state_dict(),
                'g_optimizer_state_dict': trainer.g_optimizer.state_dict(),
                'd_optimizer_state_dict': trainer.d_optimizer.state_dict(),
                'ema_shadow': trainer.ema.shadow,
                'step': step,
                'phase': phase
            }
            
            suffix = "final" if is_final else f"step_{step}"
            path = os.path.join(
                self.config.checkpoint_dir,
                f"apt_{phase}_{suffix}.pt"
            )
            
            torch.save(checkpoint, path)
            logging.info(f"Saved checkpoint to {path}")
            
    def load_checkpoint(
        self,
        trainer: 'APTTrainer',
        path: str
    ) -> Tuple[int, str]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
        trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        trainer.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        trainer.ema.shadow = checkpoint['ema_shadow']
        
        return checkpoint['step'], checkpoint['phase']

class APTTrainingLoop:
    """Main training loop for APT."""
    def __init__(
        self,
        config: APTConfig,
        trainer: APTTrainer,
        image_dataloader: Optional[DataLoader] = None,
        video_dataloader: Optional[DataLoader] = None
    ):
        self.config = config
        self.trainer = trainer
        self.image_dataloader = image_dataloader
        self.video_dataloader = video_dataloader
        
        self.logger = APTLogger(config)
        self.checkpointer = APTCheckpointer(config)
        
        # Setup distributed training if needed
        if config.world_size > 1:
            self.setup_distributed()
            
    def setup_distributed(self):
        """Setup distributed training."""
        dist.init_process_group("nccl")
        torch.cuda.set_device(self.config.local_rank)
        
        self.trainer.generator = DDP(
            self.trainer.generator,
            device_ids=[self.config.local_rank]
        )
        self.trainer.discriminator = DDP(
            self.trainer.discriminator,
            device_ids=[self.config.local_rank]
        )
        
    def train_images(self):
        """Training loop for images."""
        if not self.config.train_images:
            return
            
        self.logger.reset_metrics()
        self.trainer.set_r1_sigma(self.config.image_r1_sigma)
        
        for step in tqdm(range(self.config.image_updates)):
            batch = next(iter(self.image_dataloader))
            real_images, conditions = batch
            
            g_loss, d_loss, r1_loss = self.trainer.train_step(
                real_images,
                conditions,
                is_video=False
            )
            
            self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "image")
            
            if (step + 1) % self.config.save_interval == 0:
                self.checkpointer.save_checkpoint(
                    self.trainer,
                    step,
                    "image"
                )
                
        self.checkpointer.save_checkpoint(
            self.trainer,
            self.config.image_updates,
            "image",
            is_final=True
        )
        
    def train_videos(self):
        """Training loop for videos."""
        if not self.config.train_videos:
            return
            
        self.logger.reset_metrics()
        self.trainer.set_r1_sigma(self.config.video_r1_sigma)
        
        for step in tqdm(range(self.config.video_updates)):
            batch = next(iter(self.video_dataloader))
            real_videos, conditions = batch
            
            g_loss, d_loss, r1_loss = self.trainer.train_step(
                real_videos,
                conditions,
                is_video=True
            )
            
            self.logger.log_metrics(g_loss, d_loss, r1_loss, step, "video")
            
            if (step + 1) % self.config.save_interval == 0:
                self.checkpointer.save_checkpoint(
                    self.trainer,
                    step,
                    "video"
                )
                
        self.checkpointer.save_checkpoint(
            self.trainer,
            self.config.video_updates,
            "video",
            is_final=True
        )
        
    def train(self):
        """Run complete training process."""
        # Train on images first
        self.train_images()
        
        # Then train on videos
        self.train_videos()
        
def setup_apt_training(
    mmdit_model: nn.Module,
    config: APTConfig,
    image_dataloader: Optional[DataLoader] = None,
    video_dataloader: Optional[DataLoader] = None
) -> APTTrainingLoop:
    """Setup APT training components."""
    
    # Initialize generator and discriminator
    generator = APTGenerator(mmdit_model)
    discriminator = APTDiscriminator(
        mmdit_model,
        device=config.device,
        dtype=config.dtype
    )
    
    # Initialize R1 regularization
    r1_reg = ApproximatedR1Regularization(
        sigma=config.image_r1_sigma,
        lambda_r1=config.r1_lambda
    )
    
    # Create trainer
    trainer = APTTrainer(
        generator=generator,
        discriminator=discriminator,
        r1_reg=r1_reg,
        learning_rate=config.image_learning_rate,
        ema_decay=config.ema_decay
    )
    
    # Create training loop
    training_loop = APTTrainingLoop(
        config=config,
        trainer=trainer,
        image_dataloader=image_dataloader,
        video_dataloader=video_dataloader
    )
    
    return training_loop
