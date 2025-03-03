# Copyright 2024-2025 @johndpope All rights reserved.

import os
import sys
import warnings
from datetime import datetime
from logger import logger

warnings.filterwarnings('ignore')

import torch
import random
from omegaconf import OmegaConf
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, str2bool

# Example prompt for fallback
EXAMPLE_PROMPT = {
    "t2v-14B": {
        "prompt": "A beautiful landscape",
    },
}



def generate_batch(config, num_samples=100):
    _init_logging()
    rank = 0  # Single process for simplicity
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extract task and model config
    task = config.task if hasattr(config, 'task') else "t2v-14B"
    cfg = WAN_CONFIGS[task]
    ckpt_dir = config.ckpt_dir if hasattr(config, 'ckpt_dir') else "/path/to/checkpoints"  # Replace with your path

    # Initialize WanT2V pipeline
    logger.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=rank,
        t5_fsdp=False,  # Adjust based on config if needed
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,  # Match your earlier preference
    )

    # Generate multiple samples
    dummy_data_list = []
    base_prompt = config.sample_neg_prompt if hasattr(config, 'sample_neg_prompt') else EXAMPLE_PROMPT[task]["prompt"]
    base_seed = random.randint(0, sys.maxsize) if not hasattr(config, 'base_seed') else config.base_seed
    
    for i in range(num_samples):
        seed = base_seed + i
        prompt = f"{base_prompt} variation {i}" if i > 0 else base_prompt
        
        logger.info(f"Generating sample {i+1}/{num_samples} with prompt: {prompt}")
        video = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS["128*128"],  # Match your dummy data size
            frame_num=1,  # Single frame to match T=1
            shift=5.0,  # Default from config if not specified
            sample_solver='unipc',
            sampling_steps=50,  # Default from Wan
            guide_scale=config.cfg_scale if hasattr(config, 'cfg_scale') else 7.5,
            seed=seed,
            offload_model=True  # Default for single GPU
        )
        # video shape: [1, 16, 1, 128, 128] (B, C, T, H, W)
        dummy_data_list.append(video.squeeze(0))  # [16, 1, 128, 128]

    # Stack into [N, C, T, H, W]
    dummy_data = torch.stack(dummy_data_list, dim=0)  # [100, 16, 1, 128, 128]
    dummy_prompts = [f"{base_prompt} variation {i}" if i > 0 else base_prompt for i in range(num_samples)]
    
    return dummy_data, dummy_prompts

if __name__ == "__main__":
    # Load OmegaConf configuration
    config = OmegaConf.load("config.yaml")  # Path to your config file
    dummy_data, dummy_prompts = generate_batch(config, num_samples=100)
    
    # Save for training
    torch.save(dummy_data, "dummy_data.pt")
    torch.save(dummy_prompts, "dummy_prompts.pt")
    logger.info(f"Generated dummy_data shape: {dummy_data.shape}")