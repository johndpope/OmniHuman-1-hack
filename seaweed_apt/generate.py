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
import argparse

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

def generate_batch(config, args,num_samples=100):
    rank = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug(f"Using device: {device}")

    if not hasattr(config, 'text_len'):
        config.text_len = 512

    task = config.task if hasattr(config, 'task') else "t2v-14B"
    cfg = WAN_CONFIGS[task]
    ckpt_dir = config.ckpt_dir if hasattr(config, 'ckpt_dir') else "/path/to/checkpoints"
 # Initialize full WanT2V object
    wan_t2v = wan.WanT2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=0,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu
    )

    dummy_prompts = RANDOM_PROMPTS[:num_samples]
    base_seed = random.randint(0, sys.maxsize) if not hasattr(config, 'base_seed') else config.base_seed
    logger.debug(f"Base seed: {base_seed}, Num samples: {num_samples}")

    size = SIZE_CONFIGS["480*832"]
    target_shape = (
        wan_t2v.vae.model.z_dim, 1,
        size[0] // wan_t2v.vae_stride[1],
        size[1] // wan_t2v.vae_stride[2],
    )
    patch_size = wan_t2v.model.patch_size
    seq_len = (target_shape[1] // patch_size[0]) * \
              (target_shape[2] // patch_size[1]) * \
              (target_shape[3] // patch_size[2])
    logger.info(f"Computed seq_len: {seq_len}")
    # Precompute contexts
    wan_t2v.text_encoder.model.to(device)
    positive_contexts = [wan_t2v.text_encoder([prompt], device)[0].to(torch.float32).cpu() for prompt in dummy_prompts]
    negative_context = wan_t2v.text_encoder([config.sample_neg_prompt], device)[0].to(torch.float32).cpu()
    wan_t2v.text_encoder.model.cpu()

    # Generate noise, dummy_data (latents), and v_teacher
    logger.info("Generating noise, latents, and teacher predictions...")
    wan_t2v.model.to(device)
    noise_list = []
    dummy_data_list = []  # Latents from teacher (restored)
    v_teacher_list = []
    timestep = torch.tensor([wan_t2v.num_train_timesteps - 1], device=device, dtype=torch.float32)
    cfg_scale = 7.5  # Match training CFG scale

    for i in range(num_samples):
        seed = base_seed + i
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        noise = torch.randn(1, *target_shape, dtype=torch.float32, device=device, generator=seed_g)
        noise_list.append(noise.cpu())

        with torch.no_grad():
            context = positive_contexts[i].to(device)
            context_null = negative_context.to(device)
            noise_input = [noise[0]]
            noise_pred_cond = wan_t2v.model(noise_input, t=timestep, context=[context], seq_len=seq_len)[0]
            noise_pred_uncond = wan_t2v.model(noise_input, t=timestep, context=[context_null], seq_len=seq_len)[0]
            latent = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            dummy_data_list.append(latent.cpu())  # dummy_data as latents
            v_teacher_list.append(latent.cpu())   # v_teacher identical to latents
        torch.cuda.empty_cache()
    wan_t2v.model.cpu()
    torch.cuda.empty_cache()

    # Compile data dictionary
    dummy_data = torch.stack(dummy_data_list, dim=0)  # Shape: [100, 16, 1, 60, 104]
    noise = torch.stack(noise_list, dim=0)            # Shape: [100, 16, 1, 60, 104]
    data_dict = {
        "dummy_data": dummy_data,
        "noise": noise,  # Added for student model input
        "dummy_prompts": dummy_prompts,
        "positive_contexts": positive_contexts,
        "negative_context": negative_context,
        "v_teacher": torch.stack(v_teacher_list, dim=0)  # Shape: [100, 16, 1, 60, 104]
    }
    filename = f"dummy_data_{size[0]}x{size[1]}.pt"
    torch.save(data_dict, filename)
    logger.info(f"Generated and saved data_dict to {filename} with keys: {data_dict.keys()}")

    return data_dict


# Adjusted Dataset for Testing
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load precomputed data including positive and negative contexts
        data_dict = torch.load(data_path, map_location='cpu')
        self.samples = data_dict['dummy_data']  # Video latents
        self.positive_contexts = data_dict['positive_contexts']  # Precomputed positive contexts
        self.negative_context = data_dict['negative_context']  # Precomputed negative context
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return sample and its corresponding precomputed contexts
        sample = self.samples[idx]
        positive_context = self.positive_contexts[idx]
        # Expand negative_context to match batch size (assuming it's a single tensor)
        negative_context = self.negative_context.expand_as(positive_context)
        return sample, positive_context, negative_context

def test_text_video_dataset():
    """Test the TextVideoDataset class with the generated dummy data."""
    logger.info("Starting TextVideoDataset test...")
    
    # Define test parameters
    data_path = "dummy_data_480x832.pt"
    
    # Check if the file exists
    if not os.path.exists(data_path):
        logger.error(f"Test file {data_path} not found. Please generate dummy data first.")
        return
    
    # Initialize dataset
    dataset = TextVideoDataset(data_path)
    logger.info(f"Dataset initialized with {len(dataset)} samples.")

    # Test length
    assert len(dataset) > 0, "Dataset length should be greater than 0"
    logger.info(f"Dataset length test passed: {len(dataset)} samples")

    # Test a few samples
    for idx in range(min(3, len(dataset))):  # Test first 3 samples or less if dataset is smaller
        sample, positive_context, negative_context = dataset[idx]
        
        # Expected shapes (adjust based on your generate_batch output)
        expected_sample_shape = torch.Size([16, 1, 60, 104])  # [C, T, H, W]
        expected_context_shape = torch.Size([512, 4096])     # [L, D] for T5 embeddings
        
        # Check shapes
        assert sample.shape == expected_sample_shape, \
            f"Sample shape mismatch at index {idx}: got {sample.shape}, expected {expected_sample_shape}"
        assert positive_context.shape == expected_context_shape, \
            f"Positive context shape mismatch at index {idx}: got {positive_context.shape}, expected {expected_context_shape}"
        assert negative_context.shape == expected_context_shape, \
            f"Negative context shape mismatch at index {idx}: got {negative_context.shape}, expected {expected_context_shape}"
        
        # Log successful shape check
        logger.info(f"Sample {idx} shape test passed: sample={sample.shape}, "
                    f"positive_context={positive_context.shape}, negative_context={negative_context.shape}")

    logger.info("TextVideoDataset test completed successfully!")

# [INFO] Creating WanT2V pipeline.
# [DEBUG] Using device: cuda
# [INFO] Computed seq_len: 1560
# [INFO] Generating sample 1/1 with prompt: A futuristic city floating among the clouds at sunset
# [DEBUG] Noise shape: torch.Size([1, 16, 1, 60, 104]), dtype: torch.float32
# [DEBUG] Context shape: torch.Size([512, 4096]), dtype: torch.bfloat16
# [DEBUG] Converted context to dtype: torch.float32
# [DEBUG] Noise_pred_cond shape: torch.Size([16, 1, 60, 104]), dtype: torch.float32
# [DEBUG] Generated latent shape: torch.Size([16, 1, 60, 104]), dtype: torch.float32
# [INFO] Generated dummy_data shape: torch.Size([1, 16, 1, 60, 104]), dtype: torch.float32
# [INFO] Saved dummy_data shape: torch.Size([1, 16, 1, 60, 104]), dtype: torch.float32
if __name__ == "__main__":
    # Load OmegaConf configuration

    parser = argparse.ArgumentParser(description="Train consistency distillation for Seaweed-APT")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan T2V model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate (paper: 5e-6)")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale (paper: 7.5)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="seaweed-apt-distillation", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--config_file", type=str, default="./config.yaml", help="Path to config file")
    parser.add_argument("--save_interval", type=int, default=350, help="Save interval (paper: 350 updates)")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps for gradient accumulation")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    args = parser.parse_args()
    
    from wan.configs import t2v_14B, t2v_1_3B
    data_dict = generate_batch(t2v_1_3B, args,num_samples=100)
    
