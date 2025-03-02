import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
import librosa
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import mediapipe as mp

from wan.configs import t2v_14B
from omnihuman import OmniHumanSeaweedWan

def extract_audio_features(audio_path, num_frames, sample_rate=16000, feature_dim=1024):
    """
    Extract audio features from an audio file.
    
    Args:
        audio_path: Path to audio file
        num_frames: Number of frames to extract
        sample_rate: Audio sample rate
        feature_dim: Output feature dimension
        
    Returns:
        Audio features tensor of shape [num_frames, feature_dim]
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # In a real implementation, you would use a pretrained model like wav2vec
    # For simplicity, we'll extract mel spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Resample to match num_frames
    hop_length = len(y) // num_frames
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + hop_length, len(y))
        frame = y[start:end]
        frame_features = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=feature_dim // 8)
        frame_features = librosa.power_to_db(frame_features).flatten()
        
        # Ensure consistent dimension
        if len(frame_features) > feature_dim:
            frame_features = frame_features[:feature_dim]
        elif len(frame_features) < feature_dim:
            frame_features = np.pad(frame_features, (0, feature_dim - len(frame_features)))
            
        frames.append(frame_features)
    
    # Convert to tensor
    audio_features = torch.tensor(np.stack(frames), dtype=torch.float32)
    return audio_features

def extract_pose_heatmaps(video_path, num_frames, output_size=(64, 64), num_keypoints=33):
    """
    Extract pose heatmaps from a video file using MediaPipe.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        output_size: Size of output heatmaps
        num_keypoints: Number of keypoints to extract
        
    Returns:
        Pose heatmaps tensor of shape [num_frames, num_keypoints, output_size[0], output_size[1]]
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Extract keypoints
    heatmaps = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = pose.process(frame_rgb)
        
        # Create heatmaps
        frame_heatmaps = np.zeros((num_keypoints, output_size[0], output_size[1]), dtype=np.float32)
        
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i >= num_keypoints:
                    break
                    
                # Convert normalized coordinates to heatmap coordinates
                x = int(landmark.x * output_size[1])
                y = int(landmark.y * output_size[0])
                
                # Create Gaussian heatmap
                if 0 <= x < output_size[1] and 0 <= y < output_size[0]:
                    sigma = 2.0  # Gaussian sigma
                    for map_y in range(output_size[0]):
                        for map_x in range(output_size[1]):
                            dist = (map_x - x) ** 2 + (map_y - y) ** 2
                            frame_heatmaps[i, map_y, map_x] = np.exp(-dist / (2 * sigma ** 2))
        
        heatmaps.append(frame_heatmaps)
    
    cap.release()
    
    # Convert to tensor
    pose_heatmaps = torch.tensor(np.stack(heatmaps), dtype=torch.float32)
    return pose_heatmaps

def load_reference_image(image_path, target_size=(720, 1280)):
    """
    Load and preprocess reference image.
    
    Args:
        image_path: Path to reference image
        target_size: Target size (height, width)
        
    Returns:
        Processed image tensor of shape [3, H, W]
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Resize
    image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Convert to tensor
    image = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)
    
    # Normalize
    image = (image / 127.5) - 1.0
    
    return image

def save_video(video_tensor, output_path, fps=24):
    """
    Save video tensor to file.
    
    Args:
        video_tensor: Video tensor of shape [3, T, H, W]
        output_path: Path to save video
        fps: Frames per second
    """
    # Convert tensor to numpy
    video = video_tensor.cpu().numpy()
    
    # Transpose to [T, H, W, 3]
    video = video.transpose(1, 2, 3, 0)
    
    # Denormalize
    video = (video + 1.0) * 127.5
    video = np.clip(video, 0, 255).astype(np.uint8)
    
    # Create clip and save
    clip = ImageSequenceClip(list(video), fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps)

def main():
    parser = argparse.ArgumentParser(description="Generate videos with OmniHuman-Seaweed")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained OmniHuman-Seaweed model")
    parser.add_argument("--wan_checkpoint_dir", type=str, required=True, help="Directory with Wan model checkpoints")
    parser.add_argument("--seaweed_checkpoint", type=str, required=True, help="Path to trained Seaweed-Wan model")
    parser.add_argument("--reference_image", type=str, required=True, help="Path to reference image")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text description")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to audio file")
    parser.add_argument("--pose_video", type=str, default=None, help="Path to pose video")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=720, help="Output video height")
    parser.add_argument("--width", type=int, default=1280, help="Output video width")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = OmniHumanSeaweedWan(
        wan_config=t2v_14B,
        checkpoint_dir=args.wan_checkpoint_dir,
        seaweed_checkpoint_path=args.seaweed_checkpoint,
        num_frames=args.num_frames,
        device_id=args.device_id
    )
    
    # Load trained weights if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    
    # Load reference image
    reference_image = load_reference_image(
        args.reference_image, 
        target_size=(args.height, args.width)
    ).to(device)
    
    # Load audio features if provided
    audio_features = None
    if args.audio_path:
        audio_features = extract_audio_features(
            args.audio_path,
            args.num_frames
        ).to(device)
    
    # Load pose heatmaps if provided
    pose_heatmaps = None
    if args.pose_video:
        pose_heatmaps = extract_pose_heatmaps(
            args.pose_video,
            args.num_frames
        ).to(device)
    
    # Generate video
    with torch.no_grad():
        video = model(
            text_prompt=args.text_prompt,
            audio=audio_features,
            pose=pose_heatmaps,
            reference_image=reference_image,
            cfg_scale=args.cfg_scale,
            seed=args.seed
        )
    
    # Save video
    save_video(video, args.output_path)
    print(f"Video saved to {args.output_path}")

if __name__ == "__main__":
    main()
