import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio
from PIL import Image
import argparse
import os
from logger import logger
import wan
from wan.configs import WAN_CONFIGS



try:
    import clip
except ImportError:
    print("Warning: clip not installed. CLIP score computation skipped. Install with: pip install git+https://github.com/openai/CLIP.git")
    clip = None

# Import WAN-specific modules
from wan.modules.model import WanModel
from wan.modules.vae import WanVAE
from wan import WanT2V

# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate EMA distilled model")
parser.add_argument("--checkpoint_dir", type=str, default="../models/Wan2.1-T2V-1.3B", help="Path to pretrained model checkpoint")
parser.add_argument("--output_dir", type=str, default="./", help="Directory with EMA model")
parser.add_argument("--vae_path", type=str, default="../models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", help="Path to VAE checkpoint")
parser.add_argument("--real_video_dir", type=str, default="./real_videos", help="Directory with real videos for FVD")
parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
args = parser.parse_args()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EMA model
ema_model = WanModel.from_pretrained(args.checkpoint_dir).to(device)
ema_path = f"{args.output_dir}/ema_model_epoch_8.pt"
if not os.path.exists(ema_path):
    raise FileNotFoundError(f"EMA model file not found at {ema_path}")
ema_model.load_state_dict(torch.load(ema_path, map_location=device))
ema_model.eval()
logger.info(f"Loaded EMA model from {ema_path}")

# Load precomputed data
data_path = "dummy_data_480x832.pt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")
data_dict = torch.load(data_path, map_location="cpu")
num_samples = 10  # Evaluate 10 samples
noise = data_dict["noise"][:num_samples].to(device)
positive_contexts = [c.to(device) for c in data_dict["positive_contexts"][:num_samples]]
dummy_prompts = data_dict["dummy_prompts"][:num_samples]
v_teacher = data_dict["v_teacher"][:num_samples].to(device)

rank = 0  
# Load WanT2V and VAE
from wan.configs import t2v_14B, t2v_1_3B
config = t2v_1_3B
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
logger.debug(f"Noise shape: {noise.shape}")
logger.debug(f"v_teacher shape: {v_teacher.shape}")
logger.debug(f"Number of positive contexts: {len(positive_contexts)}")
logger.debug(f"First positive context shape: {positive_contexts[0].shape}")

# Compute sequence length
patch_size = wan_t2v.model.patch_size
vae_stride = wan_t2v.vae_stride
target_shape = (16, 1, 480 // vae_stride[1], 832 // vae_stride[2])  # [16, 1, 60, 104]
seq_len = (target_shape[1] // patch_size[0]) * \
          (target_shape[2] // patch_size[1]) * \
          (target_shape[3] // patch_size[2])  # Should be 1560
logger.info(f"Computed seq_len: {seq_len}")

# Generate EMA outputs
logger.info("Generating outputs from EMA model...")
x_latent_list = []
final_timestep = config.num_train_timesteps  # Use final timestep (e.g., 1000) for one-step generation

with torch.no_grad():
    for i in range(num_samples):
        sample_noise = noise[i:i+1]  # [1, 16, 1, 60, 104]
        sample_context = [positive_contexts[i]]  # List of [512, 4096]
        t = torch.full((1,), final_timestep, device=device, dtype=torch.float32)
        
        logger.debug(f"Processing sample {i+1}/{num_samples}")
        logger.debug(f"Sample noise shape: {sample_noise.shape}")
        logger.debug(f"Sample context shape: {sample_context[0].shape}")
        
        # One-step generation: v = model(z, T), x = z - v
        v_pred = wan_t2v.model(sample_noise, t, sample_context, seq_len)[0]  # [16, 1, 60, 104]
        sample_output = sample_noise - v_pred  # [1, 16, 1, 60, 104]
        
        logger.debug(f"v_pred shape: {v_pred.shape}")
        logger.debug(f"Sample output shape: {sample_output.shape}")
        
        x_latent_list.append(sample_output.cpu())
        torch.cuda.empty_cache()

# Stack latents correctly
x_latent = torch.cat(x_latent_list, dim=0).to(device)  # [10, 16, 1, 60, 104]
logger.info(f"Generated EMA latent outputs with shape: {x_latent.shape}")

# Decode latents to pixel space
logger.info("Decoding latents to pixel space...")
try:
    # Ensure VAE expects [B, C, T, H, W]
    if x_latent.shape[2] == 1:  # [10, 16, 1, 60, 104]
        logger.warning("Latent has T=1; generating single-frame videos. Multi-frame generation may require noise adjustment.")
    
    # Decode each sample's latent to pixel space
    x_pixel_list = wan_t2v.vae.decode([x.squeeze(0) for x in x_latent.chunk(num_samples)])  # List of [C, T, H, W]
    
    # Move to CPU and convert to NumPy
    x_pixel = torch.stack(x_pixel_list).cpu().numpy()  # [10, 3, T, 480, 832]
    logger.info(f"Decoded latents to pixel space with shape: {x_pixel.shape}")
except Exception as e:
    logger.error(f"Error decoding latents: {e}")
    import traceback
    traceback.print_exc()
    raise  # Re-raise the exception to halt execution and debug properly

# Save Generated Videos
if x_pixel is not None:  # This check is redundant now but kept for clarity
    os.makedirs("eval_videos", exist_ok=True)
    for i, video in enumerate(x_pixel):
        # Normalize from [-1, 1] to [0, 255] and adjust axes
        video_uint8 = ((video.transpose(1, 2, 3, 0) + 1) / 2 * 255).astype(np.uint8)  # [T, H, W, 3]
        output_path = f"eval_videos/eval_video_{i}.mp4"
        imageio.mimwrite(output_path, video_uint8, fps=16)  # Adjust fps as needed
        logger.info(f"Saved video {i} to {output_path}")
else:
    logger.error("No videos generated due to decoding failure.")

# 1. FVD - Compare to real videos
# def load_real_videos(directory, num_samples, shape=(16, 480, 832, 3)):
#     """Load real videos from a directory of .mp4 files."""
#     import glob
#     video_files = glob.glob(os.path.join(directory, "*.mp4"))[:num_samples]
#     if not video_files:
#         logger.warning(f"No real videos found in {directory}. FVD skipped.")
#         return None
#     real_videos = []
#     for vf in video_files:
#         reader = imageio.get_reader(vf)
#         frames = [frame for frame in reader][:shape[0]]  # Take first T frames
#         if len(frames) < shape[0]:
#             frames += [frames[-1]] * (shape[0] - len(frames))  # Pad with last frame
#         video = np.stack(frames, axis=0) / 255.0 * 2 - 1  # Normalize to [-1, 1]
#         real_videos.append(video)
#     return np.stack(real_videos, axis=0)  # [N, T, H, W, C]

# if fvd is not None:
#     real_videos = load_real_videos(args.real_video_dir, num_samples)
#     if real_videos is not None:
#         gen_videos = ((x_pixel + 1) / 2 * 255).astype(np.uint8).transpose(0, 1, 3, 4, 2)  # [N, T, H, W, C], [0, 255]
#         # Resize for I3D (224x224, 16 frames)
#         from skimage.transform import resize
#         real_videos_resized = resize(real_videos, (num_samples, 16, 224, 224, 3), preserve_range=True).astype(np.uint8)
#         gen_videos_resized = resize(gen_videos, (num_samples, 16, 224, 224, 3), preserve_range=True).astype(np.uint8)
#         real_videos_torch = torch.from_numpy(real_videos_resized).permute(0, 4, 1, 2, 3).to(device)  # [N, C, T, H, W]
#         gen_videos_torch = torch.from_numpy(gen_videos_resized).permute(0, 4, 1, 2, 3).to(device)
#         fvd_score = fvd(real_videos_torch, gen_videos_torch, device=device)
#         logger.info(f"FVD: {fvd_score.item()}")
#     else:
#         logger.info("FVD computation skipped due to missing real videos.")
# else:
#     logger.info("FVD computation skipped due to missing pytorch_fvd.")

# # 2. CLIP Score - Text-video alignment
# if clip is not None:
#     clip_model, preprocess = clip.load("ViT-B/32", device=device)
#     frames = x_pixel[:, ::4].reshape(-1, 480, 832, 3)  # Every 4th frame: [40, H, W, 3]
#     frames_processed = []
#     for frame in frames:
#         frame_img = Image.fromarray(((frame + 1) / 2 * 255).astype(np.uint8))
#         frames_processed.append(preprocess(frame_img))
#     frames_tensor = torch.stack(frames_processed).to(device)  # [40, 3, 224, 224]
#     prompts_tensor = clip.tokenize(dummy_prompts).to(device)  # [10, 77]
    
#     with torch.no_grad():
#         image_features = clip_model.encode_image(frames_tensor)  # [40, 512]
#         text_features = clip_model.encode_text(prompts_tensor)  # [10, 512]
#         # Average image features per video
#         image_features = image_features.reshape(num_samples, -1, 512).mean(dim=1)  # [10, 512]
#         clip_score = (image_features @ text_features.T).diagonal().mean().item()
#         logger.info(f"CLIP Score: {clip_score}")
# else:
#     logger.info("CLIP Score computation skipped due to missing clip.")

# # 3. PSNR/SSIM vs. Teacher
# with torch.no_grad():
#     x_teacher = noise - v_teacher  # Reconstruct teacher output
#     x_teacher_pixel = vae.decode([x.squeeze(2) for x in x_teacher.chunk(num_samples)])  # [10, T, H, W, 3]
#     x_teacher_pixel = torch.stack(x_teacher_pixel).cpu().numpy()  # [10, T, 480, 832, 3]

# psnr_scores = []
# ssim_scores = []
# for i in range(num_samples):
#     # Compute per-frame PSNR/SSIM and average
#     psnr_frame = [peak_signal_noise_ratio(x_teacher_pixel[i, j], x_pixel[i, j], data_range=2.0) 
#                   for j in range(x_pixel.shape[1])]
#     ssim_frame = [structural_similarity(x_teacher_pixel[i, j], x_pixel[i, j], multichannel=True, channel_axis=-1, data_range=2.0) 
#                   for j in range(x_pixel.shape[1])]
#     psnr_scores.append(np.mean(psnr_frame))
#     ssim_scores.append(np.mean(ssim_frame))

# psnr = np.mean(psnr_scores)
# ssim = np.mean(ssim_scores)
# logger.info(f"Avg PSNR: {psnr:.2f}, Avg SSIM: {ssim:.4f}")

# 4. Save Generated Videos
os.makedirs("eval_videos", exist_ok=True)
for i, video in enumerate(x_pixel):
    video_uint8 = ((video + 1) / 2 * 255).astype(np.uint8).transpose(0, 2, 3, 1)  # [T, H, W, C], [0, 255]
    output_path = f"eval_videos/eval_video_{i}.mp4"
    imageio.mimwrite(output_path, video_uint8, fps=16)  # Assuming 1-second clips, adjust FPS as needed
    logger.info(f"Saved video {i} to {output_path}")

logger.info("Evaluation complete.")