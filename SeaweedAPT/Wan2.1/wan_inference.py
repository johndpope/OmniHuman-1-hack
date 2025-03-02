
import argparse
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from wan.configs import t2v_14B, SIZE_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.utils import cache_video

class SeaweedWanAPTGenerator:
    def __init__(
        self,
        model_path,
        checkpoint_dir,
        device_id=0,
        multi_gpu=False,
        num_gpus=1,
    ):
        """
        Initialize the Seaweed WAN APT one-step video generator
        
        Args:
            model_path: Path to the trained one-step generator model
            checkpoint_dir: Directory with Wan checkpoints
            device_id: CUDA device ID
            multi_gpu: Whether to use multiple GPUs for inference
            num_gpus: Number of GPUs to use for parallel inference
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.multi_gpu = multi_gpu
        self.num_gpus = num_gpus
        self.config = t2v_14B
        
        # Load model components
        print(f"Loading components...")
        
        # Text encoder
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),  # Load on CPU first
            checkpoint_path=os.path.join(checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, self.config.t5_tokenizer),
        )
        
        # VAE
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, self.config.vae_checkpoint),
            device=self.device
        )
        
        # One-step generator
        self.model = WanModel.from_pretrained(checkpoint_dir)
        # Load trained weights
        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        # Move models to devices
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def generate(
        self,
        prompt,
        size=(1280, 720),
        frame_num=49,  # 2 seconds at 24fps + 1 frame
        negative_prompt="",
        seed=-1,
        save_path=None,
    ):
        """
        Generate a video using one-step inference
        
        Args:
            prompt: Text prompt for generation
            size: Video dimensions (width, height)
            frame_num: Number of frames to generate
            negative_prompt: Negative prompt for generation
            seed: Random seed (-1 for random)
            save_path: Path to save the generated video
            
        Returns:
            Generated video tensor
        """
        start_time = time.time()
        
        # Set seed
        if seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print(f"Using seed: {seed}")
        
        # Setup random generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        # Process text prompt
        print("Processing text prompt...")
        text_encode_start = time.time()
        
        # Move text encoder to device for inference
        self.text_encoder.model.to(self.device)
        
        context = self.text_encoder([prompt], self.device)
        
        if negative_prompt:
            context_null = self.text_encoder([negative_prompt], self.device)
        else:
            context_null = self.text_encoder([self.config.sample_neg_prompt], self.device)
        
        # Offload text encoder back to CPU
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        
        text_encode_time = time.time() - text_encode_start
        print(f"Text encoding completed in {text_encode_time:.2f}s")
        
        # Calculate latent dimensions
        F = frame_num
        target_shape = (
            self.vae.model.z_dim, 
            (F - 1) // self.config.vae_stride[0] + 1,
            size[1] // self.config.vae_stride[1],
            size[0] // self.config.vae_stride[2]
        )
        
        # Calculate sequence length
        seq_len = (
            (target_shape[2] * target_shape[3]) // 
            (self.config.patch_size[1] * self.config.patch_size[2]) * 
            target_shape[1]
        )
        
        # Generate latent noise
        print(f"Generating latent noise with shape {target_shape}...")
        noise = torch.randn(
            target_shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator
        ).unsqueeze(0)  # Add batch dimension
        
        # One-step inference
        print("Performing one-step inference...")
        dit_start = time.time()
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.config.param_dtype):
            latents = self.model(
                [noise], 
                t=torch.ones(1, device=self.device) * self.config.num_train_timesteps,
                context=context,
                seq_len=seq_len
            )[0]
        
        dit_time = time.time() - dit_start
        print(f"DiT inference completed in {dit_time:.2f}s")
        
        # Decode latents to video
        print("Decoding latents to video...")
        vae_start = time.time()
        
        with torch.no_grad():
            videos = self.vae.decode([latents])
        
        vae_time = time.time() - vae_start
        print(f"VAE decoding completed in {vae_time:.2f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"Total generation time: {total_time:.2f}s")
        
        # Save video if path is provided
        if save_path:
            print(f"Saving video to {save_path}...")
            save_video(videos[0], save_path)
        
        return videos[0]
    
    def __del__(self):
        # Clean up resources
        try:
            del self.model
            del self.text_encoder
            del self.vae
            torch.cuda.empty_cache()
        except:
            pass


def save_video(video_tensor, save_path, fps=24):
    """
    Save video tensor to file
    
    Args:
        video_tensor: Video tensor [C, T, H, W]
        save_path: Path to save the video
        fps: Frames per second
    """
    # Use wan utility function for saving video
    cache_video(
        video_tensor.unsqueeze(0),  # Add batch dimension
        save_file=save_path,
        fps=fps,
        normalize=True,
        value_range=(-1, 1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-step video generation with Seaweed-Wan")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained one-step generator model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with Wan checkpoints")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--frames", type=int, default=49, help="Number of frames (2s at 24fps + 1 frame)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs for inference")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Initialize generator
    generator = SeaweedWanAPTGenerator(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus,
    )
    
    # Generate video
    video = generator.generate(
        prompt=args.prompt,
        size=(args.width, args.height),
        frame_num=args.frames,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        save_path=args.output_path,
    )
    
    print(f"Video generation complete! Saved to {args.output_path}")