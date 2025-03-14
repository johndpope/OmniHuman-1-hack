# Copyright 2024-2025 @johndpope All rights reserved.

import os
import sys
import warnings
from datetime import datetime
from logger import logger
from torch.utils.data import Dataset, DataLoader
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
from distilled_trainer import TextVideoDataset


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

def generate_batch(wan_t2v, num_samples=100):



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


        # Why [16, 1, 60, 104]?
        # It’s the latent shape for one frame of a 480x832 video after VAE encoding with stride [4, 8, 8]:
        # 16: Latent channels (z_dim).
        # 60: Height ( 480÷8).
        # 104: Width (832÷8).
        # 1: Single temporal frame, with full sequence (e.g., 4 frames from 16) handled by the model or decoding.
        # What’s the 1 for?
        # It’s the temporal dimension, set to 1 to generate noise for a single latent frame per sample. This aligns with per-timestep diffusion (e.g., computing v_teacher at T-1), simplifying the batch to 100 independent frames rather than full videos.
    
    # Precompute contexts
    logger.info("Preprocessing text contexts...")
    wan_t2v.text_encoder.model.to(device)
    positive_contexts = []
    max_seq_len = config.text_len  # 512 by default
    for i, prompt in enumerate(dummy_prompts):
        logger.info(f"Processing prompt {i+1}/{num_samples}: {prompt}")
        ids, mask = wan_t2v.text_encoder.tokenizer(
            [prompt], return_mask=True, padding='max_length', max_length=max_seq_len, truncation=True
        )
        context = wan_t2v.text_encoder.model(ids.to(device), mask.to(device))
        context = context[0].to(torch.float32).cpu()  # Shape: [512, 4096]
        assert context.shape == torch.Size([max_seq_len, 4096]), \
            f"Positive context {i} shape mismatch: got {context.shape}, expected [{max_seq_len}, 4096]"
        positive_contexts.append(context)
        torch.cuda.empty_cache()

    logger.info(f"Processing negative prompt: {config.sample_neg_prompt}")
    neg_ids, neg_mask = wan_t2v.text_encoder.tokenizer(
        [config.sample_neg_prompt], return_mask=True, padding='max_length', max_length=max_seq_len, truncation=True
    )
    context_null = wan_t2v.text_encoder.model(neg_ids.to(device), neg_mask.to(device))
    negative_context = context_null[0].to(torch.float32).cpu()  # Shape: [512, 4096]
    assert negative_context.shape == torch.Size([max_seq_len, 4096]), \
        f"Negative context shape mismatch: got {negative_context.shape}, expected [{max_seq_len}, 4096]"

    # Generate noise, dummy_data (latents), and v_teacher
    logger.info("Generating noise, latents, and teacher predictions...")
    wan_t2v.model.to(device)
    noise_list = []
    dummy_data_list = []
    v_teacher_list = []
    timestep = torch.tensor([wan_t2v.num_train_timesteps - 1], device=device, dtype=torch.float32)
    cfg_scale = 7.5

    # In generate.py, within generate_batch function
    for i in range(num_samples):
        seed = base_seed + i
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        
        noise = torch.randn(
            target_shape,  # [16, 1, 60, 104]
            dtype=torch.float32,
            device=device,
            generator=seed_g
        )
        noise_list.append(noise.cpu())

        with torch.no_grad():
            context = positive_contexts[i].to(device)
            context_null = negative_context.to(device)
            noise_input = noise  # Shape: [16, 1, 60, 104]
            # Predict velocity field v at timestep T
            noise_pred_cond = wan_t2v.model(noise_input.unsqueeze(0), t=timestep, context=[context], seq_len=seq_len)[0]
            noise_pred_uncond = wan_t2v.model(noise_input.unsqueeze(0), t=timestep, context=[context_null], seq_len=seq_len)[0]
            v_teacher = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)  # Velocity field
            dummy_data_list.append(noise.cpu())  # Store noise as input
            v_teacher_list.append(v_teacher.cpu())
        torch.cuda.empty_cache()

    # Compile data dictionary with shape checks
    noise = torch.stack(noise_list, dim=0)            # Shape: [100, 16, 1, 60, 104]
    dummy_data = torch.stack(dummy_data_list, dim=0)  # Shape: [100, 16, 1, 60, 104]
    v_teacher = torch.stack(v_teacher_list, dim=0)    # Shape: [100, 16, 1, 60, 104]
    
    
    assert dummy_data.shape == torch.Size([num_samples, 16, 1, 60, 104]), \
        f"dummy_data shape mismatch: got {dummy_data.shape}, expected [{num_samples}, 16, 1, 60, 104]"
    assert noise.shape == torch.Size([num_samples, 16, 1, 60, 104]), \
        f"noise shape mismatch: got {noise.shape}, expected [{num_samples}, 16, 1, 60, 104]"
    assert v_teacher.shape == torch.Size([num_samples, 16, 1, 60, 104]), \
        f"v_teacher shape mismatch: got {v_teacher.shape}, expected [{num_samples}, 16, 1, 60, 104]"
    assert len(positive_contexts) == num_samples, \
        f"positive_contexts length mismatch: got {len(positive_contexts)}, expected {num_samples}"
    data_dict = {
        "dummy_data": dummy_data,
        "noise": noise,
        "dummy_prompts": dummy_prompts,
        "positive_contexts": positive_contexts,
        "negative_context": negative_context,
        "v_teacher": v_teacher
    }

    filename = f"dummy_data_{size[0]}x{size[1]}.pt"
    torch.save(data_dict, filename)
    logger.info(f"Generated and saved data_dict to {filename} with keys: {data_dict.keys()}")
    logger.info(f"Shape checks: dummy_data={dummy_data.shape}, noise={noise.shape}, "
                f"positive_contexts[0]={positive_contexts[0].shape}, negative_context={negative_context.shape}, "
                f"v_teacher={v_teacher.shape}")

    visualize_and_save_batch_with_vae(data_dict, wan_t2v)


    
    return data_dict

def create_dataloader(data_path, batch_size=1):
    dataset = TextVideoDataset(data_path)
    def custom_collate(batch):
        noise = torch.stack([item[0] for item in batch])  # [B, 16, 1, 60, 104]
        pos_ctx = torch.stack([item[1] for item in batch])  # [B, 512, 4096]
        v_teacher = torch.stack([item[2] for item in batch])  # [B, 16, 1, 60, 104]
        return noise, pos_ctx, v_teacher
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate), dataset

def test_dataset(data_path="dummy_data_480x832.pt"):
    """Test the TextVideoDataset and print sample shapes"""
    try:
        dataset = TextVideoDataset(data_path)
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        
        # Test getting a single item (3 items)
        noise, pos_ctx, v_teacher = dataset[0]
        logger.info(f"Sample shapes: noise={noise.shape}, pos_ctx={pos_ctx.shape}, "
                   f"v_teacher={v_teacher.shape}")
        
        # Test with dataloader
        dataloader, _ = create_dataloader(data_path, batch_size=2)
        batch = next(iter(dataloader))
        logger.info(f"Batch shapes: noise={batch[0].shape}, pos_ctx={batch[1].shape}, "
                   f"v_teacher={batch[2].shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing dataset: {e}")
        return False



def visualize_and_save_batch_with_vae(data_dict, wan_t2v, filename="sample_visualization_vae.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Get sample from data_dict
    noise = data_dict["noise"][0:1]  # Shape: [1, 16, 1, 60, 104]
    v_teacher = data_dict["v_teacher"][0:1]  # Shape: [1, 16, 1, 60, 104]
    
    # Calculate predicted x0 as: x0 = noise - v_teacher
    x0_pred = noise - v_teacher
    
    # Decode latents with VAE
    with torch.no_grad():
        print("Decoding noise...")
        noise_pixel = wan_t2v.vae.decode([noise.squeeze(0)])
        
        print("Decoding teacher prediction...")
        v_teacher_pixel = wan_t2v.vae.decode([v_teacher.squeeze(0)])
        
        print("Decoding predicted x0...")
        x0_pixel = wan_t2v.vae.decode([x0_pred.squeeze(0)])
    
    # Convert to numpy arrays for visualization (taking the first frame)
    noise_img = ((noise_pixel[0][:, 0, :, :].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    v_teacher_img = ((v_teacher_pixel[0][:, 0, :, :].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    x0_img = ((x0_pixel[0][:, 0, :, :].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the images
    axes[0].imshow(noise_img)
    axes[0].set_title("Noise")
    axes[0].axis('off')
    
    axes[1].imshow(v_teacher_img)
    axes[1].set_title("Teacher Velocity Field")
    axes[1].axis('off')
    
    axes[2].imshow(x0_img)
    axes[2].set_title("Predicted x0 (should be blurry)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Saved VAE-decoded visualization to {filename}")
    
    # Also save individual images
    plt.figure(figsize=(5, 5))
    plt.imshow(x0_img)
    plt.title("VAE-decoded x0 prediction")
    plt.axis('off')
    plt.savefig("x0_prediction.png")
    plt.close()
    
    return x0_img


def test_diffusers_pipeline(wan_t2v, prompt="A beautiful landscape with mountains and a lake", size=(480, 832), cfg_scale=7.5, seed=42):
    """
    Standalone function to test the model with diffusers pipeline using a single dummy prompt.
    
    Args:
        wan_t2v: The WanT2V model
        prompt: Text prompt to use
        size: Output image size (height, width)
        cfg_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        
    Returns:
        True if successful, False otherwise
    """
    import os
    import torch
    import numpy as np
    from PIL import Image
    
    device = wan_t2v.device
    logger.info(f"Testing diffusers pipeline with prompt: '{prompt}'")
    
    try:
        # Import diffusers components
        from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline
        
        # Create scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=wan_t2v.num_train_timesteps,
            shift=1.0,
        )
        
        # Initialize pipeline
        pipe = WanPipeline(
            tokenizer=wan_t2v.text_encoder.tokenizer,
            text_encoder=wan_t2v.text_encoder.model,
            transformer=wan_t2v.model,
            vae=wan_t2v.vae.model,
            scheduler=scheduler
        )
        
        # Move to device
        pipe = pipe.to(device)
        logger.info("Successfully initialized diffusers pipeline")
        
        # Set seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run pipeline
        logger.info("Running diffusers pipeline...")
        output = pipe(
            prompt=prompt,
            negative_prompt=wan_t2v.sample_neg_prompt,
            height=size[0], 
            width=size[1],
            num_frames=1,  # Just generate a single frame
            num_inference_steps=1,  # One-step generation
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="np",
        )
        
        # Extract and save the generated image
        image = output.frames[0][0]  # First batch, first frame
        os.makedirs("test_outputs", exist_ok=True)
        Image.fromarray((image * 255).astype(np.uint8)).save("test_outputs/diffusers_test.png")
        logger.info("Successfully generated and saved image to test_outputs/diffusers_test.png")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing diffusers pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False



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
    config = t2v_1_3B
    if not hasattr(config, 'text_len'):
        config.text_len = 512
        logger.warning("Config missing 'text_len'. Defaulting to 512.")

    # Initialize full WanT2V object
    wan_t2v = wan.WanT2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug(f"Using device: {device}")

    logger.info("Preprocessing text contexts...")
    wan_t2v.text_encoder.model.to(device)

    data_dict = generate_batch(wan_t2v, num_samples=100)
    # test_diffusers_pipeline(wan_t2v, prompt="A beautiful landscape with mountains and a lake", size=(480, 832), cfg_scale=7.5, seed=42)
   
    
    # Call this function at the end of generate_batch
    visualize_and_save_batch_with_vae(data_dict)

    success = test_dataset()
    if success:
        logger.info("Dataset test successful!")
    else:
        logger.error("Dataset test failed.")