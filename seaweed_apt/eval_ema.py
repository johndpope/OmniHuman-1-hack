import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio
from PIL import Image
import argparse
import os
from logger import logger

# Install these dependencies if not already present
try:
    from pytorch_fvd import fvd
except ImportError:
    print("Warning: pytorch_fvd not installed. FVD computation skipped. Install with: pip install pytorch-fvd")
    fvd = None

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
parser.add_argument("--vae_path", type=str, default="cache/vae_step_411000.pth", help="Path to VAE checkpoint")
parser.add_argument("--real_video_dir", type=str, default="./real_videos", help="Directory with real videos for FVD")
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

# Load WanT2V and VAE
wan_t2v = WanT2V(
    checkpoint_dir=args.checkpoint_dir,
    device_id=0,
    rank=0,
    t5_cpu=False  # Adjust based on your setup
).to(device)
vae = WanVAE(
    z_dim=16,  # Matches your latent shape [16, 1, 60, 104]
    vae_pth=args.vae_path,
    dtype=torch.float32,
    device=device
)

# Generate EMA outputs
with torch.no_grad():
    x_latent = ema_model(noise, t=torch.zeros(num_samples, device=device), context=positive_contexts, seq_len=1560)
    x_pixel = vae.decode([x.squeeze(2) for x in x_latent.chunk(num_samples)])  # [10, C, T, H, W] -> [10, T, H, W, 3]
    x_pixel = torch.stack(x_pixel).cpu().numpy()  # [10, T, 480, 832, 3], range [-1, 1]

# 1. FVD - Compare to real videos
def load_real_videos(directory, num_samples, shape=(16, 480, 832, 3)):
    """Load real videos from a directory of .mp4 files."""
    import glob
    video_files = glob.glob(os.path.join(directory, "*.mp4"))[:num_samples]
    if not video_files:
        logger.warning(f"No real videos found in {directory}. FVD skipped.")
        return None
    real_videos = []
    for vf in video_files:
        reader = imageio.get_reader(vf)
        frames = [frame for frame in reader][:shape[0]]  # Take first T frames
        if len(frames) < shape[0]:
            frames += [frames[-1]] * (shape[0] - len(frames))  # Pad with last frame
        video = np.stack(frames, axis=0) / 255.0 * 2 - 1  # Normalize to [-1, 1]
        real_videos.append(video)
    return np.stack(real_videos, axis=0)  # [N, T, H, W, C]

if fvd is not None:
    real_videos = load_real_videos(args.real_video_dir, num_samples)
    if real_videos is not None:
        gen_videos = ((x_pixel + 1) / 2 * 255).astype(np.uint8).transpose(0, 1, 3, 4, 2)  # [N, T, H, W, C], [0, 255]
        # Resize for I3D (224x224, 16 frames)
        from skimage.transform import resize
        real_videos_resized = resize(real_videos, (num_samples, 16, 224, 224, 3), preserve_range=True).astype(np.uint8)
        gen_videos_resized = resize(gen_videos, (num_samples, 16, 224, 224, 3), preserve_range=True).astype(np.uint8)
        real_videos_torch = torch.from_numpy(real_videos_resized).permute(0, 4, 1, 2, 3).to(device)  # [N, C, T, H, W]
        gen_videos_torch = torch.from_numpy(gen_videos_resized).permute(0, 4, 1, 2, 3).to(device)
        fvd_score = fvd(real_videos_torch, gen_videos_torch, device=device)
        logger.info(f"FVD: {fvd_score.item()}")
    else:
        logger.info("FVD computation skipped due to missing real videos.")
else:
    logger.info("FVD computation skipped due to missing pytorch_fvd.")

# 2. CLIP Score - Text-video alignment
if clip is not None:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    frames = x_pixel[:, ::4].reshape(-1, 480, 832, 3)  # Every 4th frame: [40, H, W, 3]
    frames_processed = []
    for frame in frames:
        frame_img = Image.fromarray(((frame + 1) / 2 * 255).astype(np.uint8))
        frames_processed.append(preprocess(frame_img))
    frames_tensor = torch.stack(frames_processed).to(device)  # [40, 3, 224, 224]
    prompts_tensor = clip.tokenize(dummy_prompts).to(device)  # [10, 77]
    
    with torch.no_grad():
        image_features = clip_model.encode_image(frames_tensor)  # [40, 512]
        text_features = clip_model.encode_text(prompts_tensor)  # [10, 512]
        # Average image features per video
        image_features = image_features.reshape(num_samples, -1, 512).mean(dim=1)  # [10, 512]
        clip_score = (image_features @ text_features.T).diagonal().mean().item()
        logger.info(f"CLIP Score: {clip_score}")
else:
    logger.info("CLIP Score computation skipped due to missing clip.")

# 3. PSNR/SSIM vs. Teacher
with torch.no_grad():
    x_teacher = noise - v_teacher  # Reconstruct teacher output
    x_teacher_pixel = vae.decode([x.squeeze(2) for x in x_teacher.chunk(num_samples)])  # [10, T, H, W, 3]
    x_teacher_pixel = torch.stack(x_teacher_pixel).cpu().numpy()  # [10, T, 480, 832, 3]

psnr_scores = []
ssim_scores = []
for i in range(num_samples):
    # Compute per-frame PSNR/SSIM and average
    psnr_frame = [peak_signal_noise_ratio(x_teacher_pixel[i, j], x_pixel[i, j], data_range=2.0) 
                  for j in range(x_pixel.shape[1])]
    ssim_frame = [structural_similarity(x_teacher_pixel[i, j], x_pixel[i, j], multichannel=True, channel_axis=-1, data_range=2.0) 
                  for j in range(x_pixel.shape[1])]
    psnr_scores.append(np.mean(psnr_frame))
    ssim_scores.append(np.mean(ssim_frame))

psnr = np.mean(psnr_scores)
ssim = np.mean(ssim_scores)
logger.info(f"Avg PSNR: {psnr:.2f}, Avg SSIM: {ssim:.4f}")

# 4. Save Generated Videos
os.makedirs("eval_videos", exist_ok=True)
for i, video in enumerate(x_pixel):
    video_uint8 = ((video + 1) / 2 * 255).astype(np.uint8).transpose(0, 2, 3, 1)  # [T, H, W, C], [0, 255]
    output_path = f"eval_videos/eval_video_{i}.mp4"
    imageio.mimwrite(output_path, video_uint8, fps=16)  # Assuming 1-second clips, adjust FPS as needed
    logger.info(f"Saved video {i} to {output_path}")

logger.info("Evaluation complete.")