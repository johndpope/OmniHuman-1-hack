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
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, str2bool

# Example prompt for fallback
EXAMPLE_PROMPT = {
    "t2v-14B": {
        "prompt": "A beautiful landscape",
    },
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}

# 100 random prompts
RANDOM_PROMPTS = [
    "A futuristic city floating among the clouds at sunset",
    "A dragon soaring over a misty mountain range",
    "A cheerful robot dancing in a neon-lit disco",
    "A serene beach with glowing bioluminescent waves at night",
    "A pack of wolves howling under a full moon in a snowy forest",
    "A steampunk airship gliding through a stormy sky",
    "A colorful parade of animated toys marching through a quiet village",
    "A lone astronaut exploring a vibrant alien jungle",
    "A medieval knight battling a giant serpent in a dark castle",
    "A whimsical garden where flowers sing in harmony",
    "A cyberpunk street race with holographic cars",
    "A peaceful meadow where deer graze under a rainbow",
    "A mysterious figure painting the sky with auroras",
    "A bustling marketplace on a desert planet with two suns",
    "A pirate ship sailing through a sea of swirling galaxies",
    "A child flying a kite shaped like a phoenix in a golden field",
    "A group of penguins sliding down an icy cliff into the ocean",
    "A magical library where books float and whisper secrets",
    "A volcanic eruption painting the sky with fiery reds and oranges",
    "A retro diner on wheels cruising through a starry desert",
    "A flock of glowing butterflies illuminating a dark forest",
    "A samurai dueling a shadow warrior in a bamboo grove",
    "A hot air balloon drifting over a patchwork of colorful fields",
    "A futuristic train speeding through a tunnel of light",
    "A mermaid singing atop a coral reef under shimmering waters",
    "A time traveler stepping out of a portal into ancient Rome",
    "A jazz band of skeletons performing in a foggy graveyard",
    "A giant turtle carrying a lush island on its back",
    "A storm chaser racing toward a towering tornado",
    "A whimsical tea party hosted by talking rabbits in a forest",
    "A spaceship crash-landing on a glowing crystal planet",
    "A painter bringing a canvas to life with dancing figures",
    "A herd of elephants bathing in a sparkling river at dawn",
    "A ghostly galleon drifting through a foggy harbor",
    "A futuristic gladiator arena with robots battling in zero gravity",
    "A cherry blossom tree shedding petals over a quiet pond",
    "A lone wolf silhouetted against a blood-red sunset",
    "A carnival with spinning lights and laughing clowns at night",
    "A diver exploring a sunken city filled with glowing fish",
    "A wizard casting spells that turn raindrops into gemstones",
    "A steam locomotive racing across a golden prairie",
    "A flock of cranes flying over a misty Japanese village",
    "A futuristic farmer tending crops in a vertical greenhouse",
    "A lion roaring atop a cliff overlooking a vast savanna",
    "A ballet dancer performing on a stage of floating lilies",
    "A cybernetic owl hunting in a neon jungle",
    "A Viking longship navigating icy fjords under the northern lights",
    "A whimsical clock tower where time flows backward",
    "A surfer riding a massive wave under a stormy sky",
    "A scientist mixing potions that explode into colorful smoke",
    "A pack of wild horses galloping through a dusty canyon",
    "A lantern festival lighting up a serene river at dusk",
    "A futuristic battlefield with drones clashing in the sky",
    "A polar bear fishing on an ice floe in the Arctic",
    "A painter’s brushstrokes animating a field of sunflowers",
    "A submarine exploring a glowing underwater volcano",
    "A fairy weaving a tapestry of stars in a moonlit glade",
    "A train of camels crossing a shimmering desert mirage",
    "A futuristic concert with holographic musicians",
    "A pack of foxes playing in a snowy meadow at twilight",
    "A spaceship docking at a bustling space station",
    "A knight riding a griffin through a stormy battlefield",
    "A serene lake reflecting a sky full of shooting stars",
    "A robot gardener tending a forest of metal trees",
    "A pirate captain steering through a whirlpool of gold coins",
    "A flock of eagles soaring above jagged mountain peaks",
    "A magical carousel spinning in a misty park",
    "A futuristic cityscape with flying taxis at dawn",
    "A bear fishing in a rushing river surrounded by autumn leaves",
    "A witch brewing a storm in a bubbling cauldron",
    "A herd of bison charging across a windswept plain",
    "A glowing jellyfish ballet in a dark ocean abyss",
    "A futuristic explorer mapping a planet of floating islands",
    "A lone guitarist strumming under a starry desert sky",
    "A dragon boat racing through a misty harbor",
    "A pack of hyenas laughing in a moonlit savanna",
    "A steampunk inventor unveiling a flying machine",
    "A serene temple garden with koi swimming in a pond",
    "A futuristic marathon with runners in exosuits",
    "A wolf pack hunting under a crimson aurora",
    "A pirate treasure chest opening to reveal glowing jewels",
    "A flock of parrots flying over a tropical rainforest",
    "A magical blacksmith forging a sword from starlight",
    "A futuristic library with floating holographic books",
    "A bear cub playing in a field of wildflowers",
    "A ninja leaping across rooftops under a crescent moon",
    "A hot springs oasis in a snowy mountain valley",
    "A futuristic zoo with robotic animals",
    "A fisherman casting a net in a glowing lagoon",
    "A witch’s broom sweeping through a stormy sky",
    "A herd of deer leaping through a misty forest",
    "A futuristic chef cooking with levitating ingredients",
    "A lone eagle perched on a cliff overlooking the ocean",
    "A magical forest where trees glow with bioluminescent light",
    "A steampunk submarine diving into a coral reef",
    "A pack of sled dogs racing across a frozen tundra",
    "A futuristic dance party with glowing floors",
    "A serene waterfall cascading into a crystal pool",
    "A dragon rider soaring through a canyon of clouds",
    "A whimsical train journey through a land of candy",
]

def generate_batch(config, num_samples=100):
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
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    # Generate multiple samples with random prompts
    dummy_data_list = []
    dummy_prompts = RANDOM_PROMPTS[:num_samples]  # Use first num_samples prompts
    base_seed = random.randint(0, sys.maxsize) if not hasattr(config, 'base_seed') else config.base_seed
    
    for i, prompt in enumerate(dummy_prompts):
        seed = base_seed + i
        
        logger.info(f"Generating sample {i+1}/{num_samples} with prompt: {prompt}")
        video = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS["480*832"],  # Grok thinks 128*128 will work.... Match your dummy data size [N, 16, 1, 128, 128]
            frame_num=1,  # Single frame to match T=1
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=config.cfg_scale if hasattr(config, 'cfg_scale') else 7.5,
            seed=seed,
            offload_model=True
        )
        # video shape: [1, 16, 1, 128, 128] (B, C, T, H, W)
        dummy_data_list.append(video.squeeze(0))  # [16, 1, 128, 128]

    # Stack into [N, C, T, H, W]
    dummy_data = torch.stack(dummy_data_list, dim=0)  # [100, 16, 1, 128, 128]
    
    return dummy_data, dummy_prompts

if __name__ == "__main__":
    # Load OmegaConf configuration
    config = OmegaConf.load("config.yaml")  # Path to your config file
    dummy_data, dummy_prompts = generate_batch(config, num_samples=100)
    
    # Save for training
    torch.save(dummy_data, "dummy_data.pt")
    torch.save(dummy_prompts, "dummy_prompts.pt")
    logger.info(f"Generated dummy_data shape: {dummy_data.shape}")