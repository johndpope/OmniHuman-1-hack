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
import torchaudio
import mediapipe as mp
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import wandb
from keypoint_processor import SapiensKeypointProcessor
from datetime import datetime
import hashlib
import random
import subprocess
import traceback
from video_tracker import VideoEventData, VideoEvent, ProblematicVideoTracker



class OmniHumanDataset(Dataset):
    """Dataset for OmniHuman training with Sapiens 308 keypoints and mixed condition support."""
    
    def __init__(
        self,
        config: DictConfig,
        condition_ratios: Dict[str, float]
    ):
        """Initialize the OmniHuman dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
            condition_ratios: Ratios for different condition modalities
        """
        super().__init__()
        
        # Extract parameters from config
        data_config = config.get('data', {})
        
        # Dataset location and properties
        self.data_dir = Path(data_config.get('data_dir', './data'))
        self.condition_ratios = condition_ratios
        self.num_frames = config.get('num_frames', 49)
        self.frame_size = data_config.get('frame_size', (256, 256))
        self.num_keypoints = config.get('num_keypoints', 308)
        self.heatmap_size = data_config.get('heatmap_size', (64, 64))
        self.sigma = data_config.get('sigma', 2.0)
        
        # Audio settings
        self.audio_sampling_rate = data_config.get('audio_sampling_rate', 16000)
        self.audio_features_dim = data_config.get('audio_features_dim', 1024)
        
        # Quality thresholds
        self.min_quality_score = data_config.get('min_quality_score', 0.7)
        self.min_motion_score = data_config.get('min_motion_score', 0.5)
        
        # Caching settings
        self.cache_audio = config.get('cache_audio', True)
        self.cache_keypoints = config.get('cache_keypoints', True)
        self.preextract_audio = config.get('preextract_audio', True)
        self.preextract_keypoints = config.get('preextract_keypoints', True)
        self.max_videos = config.get('max_videos', None)
        
        # Set up cache directories
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_cache_dir = self.cache_dir / "audio"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.keypoints_cache_dir = self.cache_dir / "keypoints"
        self.keypoints_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Keypoint processor settings
        sapiens_config = config.get('sapiens', {})
        self.sapiens_checkpoints_dir = sapiens_config.get('checkpoints_dir')
        self.sapiens_model_name = sapiens_config.get('model_name', '1b')
        self.sapiens_detection_config = sapiens_config.get('detection_config')
        self.sapiens_detection_checkpoint = sapiens_config.get('detection_checkpoint')
        
        # Initialize problematic videos tracker
        self.tracker = ProblematicVideoTracker(self.cache_dir / "bad_videos")
        
        # Setup transforms
        self.transform = self._setup_transforms(data_config)
        self.reference_transform = self._setup_reference_transforms(data_config)
            
        # Initialize keypoint processor
        self.keypoint_processor = self._init_keypoint_processor()
            
        # Get all videos and load status
        self.video_paths, self.audio_status = self._get_video_paths()
        
        # Pre-extract audio if requested
        if self.preextract_audio:
            self._preextract_all_audio()
            
        # Pre-extract keypoints if requested
        if self.preextract_keypoints and self.keypoint_processor:
            self._preextract_all_keypoints()
            
        # Load dataset annotations
        self.annotations = self._load_annotations()
        
        # Filter data based on quality and condition availability
        self.filtered_data = self._filter_data()
        
        logger.info(f"Loaded dataset with {len(self.filtered_data)} samples")
    
    def _get_video_paths(self) -> Tuple[List[str], Dict[str, Dict]]:
        """Get all video paths and check audio status."""
        # Find all video files
        all_videos = [str(f) for f in self.data_dir.glob("**/*.mp4")]
        logger.info(f"Found {len(all_videos)} video files")
        
        # Sample videos if needed
        if self.max_videos and len(all_videos) > self.max_videos:
            random.seed(42)  # For reproducibility
            videos = random.sample(all_videos, self.max_videos)
            logger.info(f"Sampled {self.max_videos} videos")
        else:
            videos = all_videos
            
        # Check audio status
        audio_status = self._check_audio_status(videos)
        
        # Filter out videos without audio
        valid_videos = [v for v in videos if audio_status.get(v, {}).get('has_audio', False)]
        logger.info(f"Found {len(valid_videos)} videos with valid audio")
        
        return valid_videos, audio_status
    
    def _check_audio_status(self, video_paths: List[str]) -> Dict[str, Dict]:
        """Check which videos have audio streams with intelligent caching."""
        audio_status = {}
        audio_status_file = self.audio_cache_dir / "audio_status.json"
        
        # Load cached status if available
        if audio_status_file.exists():
            with open(audio_status_file, 'r') as f:
                cached_status = json.load(f)
                logger.info(f"Loaded cached audio status for {len(cached_status)} videos")
        else:
            cached_status = {}
            logger.info("No cached audio status found")
        
        # For each video:
        # 1. If in cache and has_audio is True and has_cache is True - use cached result
        # 2. If new video or cache says has_audio but no cache file - check it
        videos_to_check = []
        for video_path in video_paths:
            if video_path in cached_status:
                video_status = cached_status[video_path]
                audio_path = self._get_audio_path(video_path)
                
                # If cache says it has audio and the file exists, use cached result
                if (video_status.get('has_audio', False) and 
                    video_status.get('has_cache', False) and 
                    audio_path.exists()):
                    audio_status[video_path] = video_status
                    continue
            
            # Need to check this video
            videos_to_check.append(video_path)
        
        # Report stats
        logger.info(f"Using {len(audio_status)} cached audio results")
        logger.info(f"Need to check {len(videos_to_check)} videos")
        
        # Only check videos that weren't in cache or need rechecking
        if videos_to_check:
            logger.info("Checking audio for videos not in cache...")
            for video_path in tqdm(videos_to_check, desc="Checking audio"):
                try:
                    # Check for existing audio file
                    audio_path = self._get_audio_path(video_path)
                    has_cache = audio_path.exists()
                    
                    # Check for audio stream
                    command = [
                        'ffprobe',
                        '-loglevel', 'error',
                        '-show_streams',
                        '-select_streams', 'a',
                        '-show_entries', 'stream=codec_type',
                        '-of', 'json',
                        video_path
                    ]
                    
                    result = subprocess.run(command, capture_output=True, text=True)
                    has_audio = False
                    
                    if result.returncode == 0:
                        data = json.loads(result.stdout)
                        has_audio = bool(data.get('streams', []))
                        
                        if not has_audio:
                            # Track problematic video
                            self.tracker.dispatch(VideoEventData(
                                video_path=video_path,
                                event_type=VideoEvent.NO_AUDIO,
                                details={"error": "No audio stream found in video"}
                            ))
                    else:
                        # Track error
                        self.tracker.dispatch(VideoEventData(
                            video_path=video_path,
                            event_type=VideoEvent.PROCESSING_ERROR,
                            details={
                                "error": "FFprobe error checking audio",
                                "stderr": result.stderr
                            }
                        ))
                    
                    audio_status[video_path] = {
                        'has_audio': has_audio,
                        'has_cache': has_cache,
                        'last_checked': str(datetime.now())
                    }
                    
                except Exception as e:
                    logger.error(f"Error checking {video_path}: {str(e)}")
                    # Track error
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.PROCESSING_ERROR,
                        details={
                            "error": f"Audio check error: {str(e)}",
                            "traceback": traceback.format_exc()
                        }
                    ))
                    audio_status[video_path] = {
                        'has_audio': False,
                        'has_cache': False,
                        'error': str(e),
                        'last_checked': str(datetime.now())
                    }
                    
        # Save updated status
        with open(audio_status_file, 'w') as f:
            json.dump({**cached_status, **audio_status}, f, indent=2)
                
        return audio_status
    
    def _setup_transforms(self, data_config: DictConfig) -> T.Compose:
        """Set up image transforms based on config."""
        if hasattr(data_config, 'transform') and data_config.transform:
            # Use custom transform from config if provided
            return data_config.transform
        else:
            # Use default transform
            return T.Compose([
                T.Resize(self.frame_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _setup_reference_transforms(self, data_config: DictConfig) -> T.Compose:
        """Set up reference image transforms based on config."""
        # No random flip for reference to maintain consistency
        return T.Compose([
            T.Resize(self.frame_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _init_keypoint_processor(self) -> Optional[SapiensKeypointProcessor]:
        """Initialize Sapiens keypoint processor."""
        if self.sapiens_checkpoints_dir:
            try:
                processor = SapiensKeypointProcessor(
                    checkpoints_dir=self.sapiens_checkpoints_dir,
                    model_name=self.sapiens_model_name,
                    detection_config=self.sapiens_detection_config,
                    detection_checkpoint=self.sapiens_detection_checkpoint,
                    heatmap_size=self.heatmap_size,
                )
                logger.info(f"Initialized Sapiens keypoint processor with model {self.sapiens_model_name}")
                return processor
            except Exception as e:
                logger.error(f"Failed to initialize Sapiens keypoint processor: {e}")
                return None
        else:
            logger.warning("No Sapiens checkpoints directory provided, pose extraction will not be available")
            return None
            
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations with automatic generation if needed."""
        annotation_file = self.data_dir / 'annotations.json'
        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            # Generate minimal annotations from video structure
            return self._generate_annotations()
            
        with open(annotation_file, 'r') as f:
            return json.load(f)
    
    def _generate_annotations(self) -> List[Dict]:
        """Generate minimal annotations from video files."""
        annotations = []
        
        for video_path in self.video_paths:
            # Skip videos without audio
            if not self.audio_status.get(video_path, {}).get('has_audio', False):
                continue
                
            # Create relative path from data_dir
            video_path_obj = Path(video_path)
            relative_path = video_path_obj.relative_to(self.data_dir)
            parent_dir = relative_path.parent
            
            # Get video info using opencv
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    logger.warning(f"Could not open video: {video_path}")
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                # Skip if video is too short
                if total_frames < self.num_frames:
                    logger.warning(f"Video too short: {video_path} ({total_frames} frames)")
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.VIDEO_TOO_SHORT,
                        details={
                            "total_frames": total_frames,
                            "min_frames": self.num_frames
                        }
                    ))
                    continue
                    
                # Create annotation
                annotation = {
                    'id': str(len(annotations)),
                    'relative_path': str(parent_dir),
                    'duration': duration,
                    'fps': fps,
                    'total_frames': total_frames,
                    'quality_score': 1.0,  # Default quality
                    'motion_score': 1.0,   # Default motion
                    'has_audio': True,     # We filtered for audio
                    'has_pose': self.keypoint_processor is not None,
                    'has_text': False,     # No text by default
                    'has_reference': True, # Use first frame as reference
                    'video_name': video_path_obj.name
                }
                
                annotations.append(annotation)
                
            except Exception as e:
                logger.error(f"Error generating annotation for {video_path}: {str(e)}")
                continue
        
        # Save generated annotations
        annotation_file = self.data_dir / 'annotations.json'
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
            
        logger.info(f"Generated {len(annotations)} annotations")
        return annotations
            
    def _filter_data(self) -> List[Dict]:
        """Filter data based on quality and available conditions."""
        filtered = []
        
        for item in tqdm(self.annotations, desc="Filtering data"):
            # Skip if quality is too low
            if item.get('quality_score', 0) < self.min_quality_score:
                continue
                
            # Skip if motion is too low
            if item.get('motion_score', 0) < self.min_motion_score:
                continue
                
            # Check if required conditions are available
            valid_conditions = True
            for cond, ratio in self.condition_ratios.items():
                if ratio > 0 and not item.get(f'has_{cond}', False):
                    valid_conditions = False
                    break
                    
            if valid_conditions:
                filtered.append(item)
                
        logger.info(f"Filtered {len(self.annotations)} samples to {len(filtered)} based on quality and conditions")
        return filtered
        
    def _get_audio_path(self, video_path: str) -> Path:
        """Get the path where the audio file should be stored."""
        video_path = Path(video_path)
        if self.cache_audio:
            # Create a unique filename based on video path
            video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
            return self.audio_cache_dir / f"{video_hash}.wav"
        else:
            return video_path.with_suffix('.wav')
            
    def _get_keypoints_path(self, video_path: str, frame_idx: int) -> Path:
        """Get the path where the keypoints file should be stored."""
        video_path = Path(video_path)
        if self.cache_keypoints:
            # Create a unique filename based on video path and frame index
            video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
            return self.keypoints_cache_dir / f"{video_hash}_{frame_idx:06d}.npy"
        else:
            keypoints_dir = video_path.parent / "keypoints"
            keypoints_dir.mkdir(exist_ok=True)
            return keypoints_dir / f"frame_{frame_idx:06d}.npy"
    
    def _extract_audio(self, video_path: str) -> None:
        """Extract audio from video file using ffmpeg."""
        try:
            # Skip if video has no audio
            if not self.audio_status.get(video_path, {}).get('has_audio', False):
                logger.info(f"Skipping audio extraction for {video_path} - no audio stream")
                return
                
            audio_path = self._get_audio_path(video_path)
            
            # Skip if audio file already exists
            if audio_path.exists():
                return
                
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sampling rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                str(audio_path)
            ]
            
            # Create directory if it doesn't exist
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting audio from {video_path} to {audio_path}")
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
                
            if not audio_path.exists():
                raise RuntimeError(f"FFmpeg completed but audio file not created at {audio_path}")
                
            logger.info(f"Successfully extracted audio to {audio_path}")
            
            # Update audio status
            self.audio_status[video_path]['has_cache'] = True
            
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {str(e)}")
            self.tracker.dispatch(VideoEventData(
                video_path=video_path,
                event_type=VideoEvent.PROCESSING_ERROR,
                details={
                    "error": f"Audio extraction error: {str(e)}",
                    "traceback": traceback.format_exc()
                }
            ))
            raise
            
    def _preextract_all_audio(self):
        """Pre-extract audio for all videos with audio streams."""
        videos_with_audio = [v for v in self.video_paths 
                            if self.audio_status.get(v, {}).get('has_audio', False)]
        
        if not videos_with_audio:
            logger.warning("No videos with audio found for pre-extraction")
            return
            
        logger.info(f"Pre-extracting audio for {len(videos_with_audio)} videos")
        
        for video_path in tqdm(videos_with_audio, desc="Extracting audio"):
            try:
                audio_path = self._get_audio_path(video_path)
                if not audio_path.exists():
                    self._extract_audio(video_path)
            except Exception as e:
                logger.error(f"Failed to extract audio for {video_path}: {str(e)}")
                continue
                
        # Save updated audio status
        audio_status_file = self.audio_cache_dir / "audio_status.json"
        with open(audio_status_file, 'w') as f:
            json.dump(self.audio_status, f, indent=2)
    
    def _extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract keypoints from frame."""
        if not self.keypoint_processor:
            return None
            
        try:
            # Convert to correct format if needed
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy()
                
            # Scale to 0-255 range if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
                
            # Ensure correct color format (RGB)
            if frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif frame.shape[2] == 3:  # Assume RGB
                pass
            elif frame.shape[2] == 1:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Unexpected frame shape: {frame.shape}")
                    
            # Extract keypoints
            keypoints = self.keypoint_processor.extract_keypoints(frame)
            
            # Convert dictionary to numpy array if needed
            if isinstance(keypoints, dict):
                keypoints_array = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                for i, name in enumerate(keypoints):
                    if i < self.num_keypoints:
                        keypoints_array[i] = keypoints[name]
                return keypoints_array
                
            return keypoints
            
        except Exception as e:
            logger.error(f"Error extracting keypoints: {str(e)}")
            return np.zeros((self.num_keypoints, 3), dtype=np.float32)

            
            
    def _preextract_all_keypoints(self):
        """Pre-extract keypoints for all videos."""
        if not self.keypoint_processor:
            logger.warning("No keypoint processor available for pre-extraction")
            return
            
        logger.info(f"Pre-extracting keypoints for {len(self.video_paths)} videos")
        
        for video_path in tqdm(self.video_paths, desc="Extracting keypoints"):
            try:
                # Check and create video frames directory structure
                video_path_obj = Path(video_path)
                frames_dir = video_path_obj.parent / "frames"
                frames_dir.mkdir(exist_ok=True)
                
                # Open video
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    logger.warning(f"Could not open video: {video_path}")
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Skip if video is too short
                if total_frames < self.num_frames:
                    logger.warning(f"Video too short: {video_path} ({total_frames} frames)")
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.VIDEO_TOO_SHORT,
                        details={
                            "total_frames": total_frames,
                            "min_frames": self.num_frames
                        }
                    ))
                    cap.release()
                    continue
                
                # Sample frames evenly
                sample_indices = np.linspace(0, total_frames-1, min(self.num_frames, total_frames), dtype=int)
                
                # Extract keypoints for each sampled frame
                valid_keypoints = 0
                for i, frame_idx in enumerate(sample_indices):
                    # Check if keypoints already cached
                    keypoints_path = self._get_keypoints_path(video_path, frame_idx)
                    if keypoints_path.exists():
                        valid_keypoints += 1
                        continue
                        
                    # Read frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                        continue
                        
                    # Convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Extract keypoints
                    keypoints = self._extract_keypoints(frame)
                    if keypoints is None:
                        logger.warning(f"Failed to extract keypoints for frame {frame_idx} from {video_path}")
                        self.tracker.dispatch(VideoEventData(
                            video_path=video_path,
                            event_type=VideoEvent.LANDMARK_DETECTION_FAILED,
                            details={
                                "frame_idx": int(frame_idx),
                                "error": "Failed to extract keypoints"
                            }
                        ))
                        continue
                        
                    # Save keypoints
                    keypoints_path.parent.mkdir(exist_ok=True)
                    np.save(keypoints_path, keypoints)
                    valid_keypoints += 1
                    
                    # Save frame for reference
                    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                cap.release()
                
                # Check if we have enough valid keypoints
                if valid_keypoints < self.num_frames * 0.5:  # At least 50% success rate
                    logger.warning(f"Too few valid keypoints for {video_path}: {valid_keypoints}/{self.num_frames}")
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.INVALID_FRAMES,
                        details={
                            "valid_keypoints": valid_keypoints,
                            "total_frames": self.num_frames,
                            "error": "Too few valid keypoints"
                        }
                    ))
                    
            except Exception as e:
                logger.error(f"Failed to extract keypoints for {video_path}: {str(e)}")
                self.tracker.dispatch(VideoEventData(
                    video_path=video_path,
                    event_type=VideoEvent.PROCESSING_ERROR,
                    details={
                        "error": f"Keypoint extraction error: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
                ))
                continue
    
    def _load_frames(self, video_path: Path, start_idx: int = 0) -> torch.Tensor:
        """Load and preprocess video frames."""
        # Try to load from frames directory first
        frames_dir = video_path / 'frames'
        if frames_dir.exists():
            frame_files = sorted(list(frames_dir.glob('*.jpg')))
        else:
            # If no frames directory, use original video
            video_file = list(video_path.glob('*.mp4'))
            if not video_file:
                logger.warning(f"No video file found in {video_path}")
                return torch.zeros((self.num_frames, 3, *self.frame_size))
                
            # Extract frames from video
            return self._extract_frames_from_video(video_file[0], start_idx)
        
        if len(frame_files) == 0:
            logger.warning(f"No frames found in {frames_dir}")
            # Return zeros as fallback
            return torch.zeros((self.num_frames, 3, *self.frame_size))
        
        # Handle case with fewer frames than needed
        if len(frame_files) < self.num_frames:
            # If we have fewer frames than needed, repeat the last frame
            frame_files = frame_files + [frame_files[-1]] * (self.num_frames - len(frame_files))
        
        # Sample frames if we have more than needed
        if len(frame_files) > self.num_frames:
            # Ensure we start from start_idx
            available_frames = len(frame_files) - start_idx
            if available_frames < self.num_frames:
                logger.warning(f"Not enough frames from start_idx {start_idx}: have {available_frames}, need {self.num_frames}")
                # Fall back to taking all available frames
                frame_indices = range(len(frame_files))[:self.num_frames]
            else:
                frame_indices = range(start_idx, start_idx + self.num_frames)
            
            frame_files = [frame_files[i] for i in frame_indices]
        
        # Load and process frames
        frames = []
        for frame_file in frame_files[:self.num_frames]:
            try:
                frame = Image.open(frame_file).convert('RGB')
                frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                logger.error(f"Error loading frame {frame_file}: {str(e)}")
                # Add a blank frame as fallback
                frames.append(torch.zeros((3, *self.frame_size)))
            
        return torch.stack(frames)  # [T, C, H, W]
        
    def _extract_frames_from_video(self, video_path: Path, start_idx: int = 0) -> torch.Tensor:
        """Extract frames directly from video file."""
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return torch.zeros((self.num_frames, 3, *self.frame_size))
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate start index
            if start_idx >= total_frames:
                logger.warning(f"Start index {start_idx} exceeds total frames {total_frames}")
                return torch.zeros((self.num_frames, 3, *self.frame_size))
                
            # Set position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            
            # Read frames
            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    # If we run out of frames, add zeros
                    frames.append(torch.zeros((3, *self.frame_size)))
                    continue
                    
                # Convert to RGB and transform
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame)
                frame_tensor = self.transform(frame_pil)
                frames.append(frame_tensor)
                
            cap.release()
            
            # Stack frames
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return torch.zeros((self.num_frames, 3, *self.frame_size))
                
            return torch.stack(frames)
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return torch.zeros((self.num_frames, 3, *self.frame_size))
            
    def _load_audio(self, video_path: Path) -> torch.Tensor:
        """Load and process audio features."""
        try:
            # Get audio path
            audio_path = self._get_audio_path(str(video_path.parent / video_path.name))
            
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return torch.zeros((self.num_frames, self.audio_features_dim))
                
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sample_rate != self.audio_sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.audio_sampling_rate
                )
                waveform = resampler(waveform)
                
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Get sample length that corresponds to one video frame
            # For a 30fps video with 16kHz audio, this is 533 samples per frame
            samples_per_frame = self.audio_sampling_rate / 30.0  # Assuming 30fps
            
            # Extract features for each frame
            features = []
            for i in range(self.num_frames):
                start_sample = int(i * samples_per_frame)
                end_sample = int((i + 1) * samples_per_frame)
                
                if end_sample > waveform.shape[1]:
                    # Pad with zeros if we reach the end
                    frame_audio = torch.zeros(1, int(samples_per_frame))
                    if start_sample < waveform.shape[1]:
                        frame_audio[:, :waveform.shape[1]-start_sample] = waveform[:, start_sample:]
                else:
                    frame_audio = waveform[:, start_sample:end_sample]
                
                # Here you would typically extract features using a model like Wav2Vec
                # For simplicity, we'll use a random projection for now
                feature = torch.randn(self.audio_features_dim)
                features.append(feature)
                
            # Stack features
            return torch.stack(features)
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {str(e)}")
            return torch.zeros((self.num_frames, self.audio_features_dim))
            
    def _load_text_embedding(self, text_embedding_path: Path) -> torch.Tensor:
        """Load text embedding."""
        if not text_embedding_path.exists():
            logger.warning(f"Text embedding file not found: {text_embedding_path}")
            return torch.zeros(1, 768)  # Default T5 dimension
            
        try:
            embedding = torch.load(text_embedding_path)
            return embedding
        except Exception as e:
            logger.error(f"Error loading text embedding {text_embedding_path}: {str(e)}")
            return torch.zeros(1, 768)
    
    def _generate_heatmaps(self, keypoints_list: List[np.ndarray]) -> torch.Tensor:
        """Generate heatmaps from keypoints.
        
        Args:
            keypoints_list: List of keypoint arrays [T, K, 3]
            
        Returns:
            Heatmaps tensor [T, K, H, W]
        """
        T = len(keypoints_list)
        K = self.num_keypoints
        H, W = self.heatmap_size
        heatmaps = torch.zeros((T, K, H, W), dtype=torch.float32)
        
        # Either use the keypoint processor's heatmap generator or generate our own
        if self.keypoint_processor:
            for t, keypoints in enumerate(keypoints_list):
                try:
                    heatmap = self.keypoint_processor.generate_heatmaps(keypoints)
                    heatmaps[t] = torch.from_numpy(heatmap)
                except Exception as e:
                    logger.error(f"Error generating heatmap for frame {t}: {str(e)}")
                    # Leave zeros for this frame
        else:
            # Manual heatmap generation
            for t, keypoints in enumerate(keypoints_list):
                for k in range(K):
                    if keypoints[k, 2] > 0.1:  # Confidence threshold
                        x, y = keypoints[k, :2]
                        x_scaled = int(x * W)
                        y_scaled = int(y * H)
                        
                        if 0 <= x_scaled < W and 0 <= y_scaled < H:
                            # Create 2D Gaussian
                            grid_y = torch.arange(H, dtype=torch.float32)
                            grid_x = torch.arange(W, dtype=torch.float32)
                            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                            
                            dist_squared = (grid_x - x_scaled)**2 + (grid_y - y_scaled)**2
                            exponent = -dist_squared / (2 * self.sigma**2)
                            heatmap = torch.exp(exponent)
                            heatmaps[t, k] = heatmap
                            
        return heatmaps
    
    def _load_keypoints_from_cache(self, video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
        """Load keypoints from cached files for specified frames."""
        keypoints_list = []
        
        for frame_idx in frame_indices:
            keypoints_path = self._get_keypoints_path(video_path, frame_idx)
            
            try:
                if keypoints_path.exists():
                    keypoints = np.load(keypoints_path)
                    
                    # Ensure correct shape
                    if keypoints.shape[0] != self.num_keypoints:
                        logger.warning(f"Keypoints in {keypoints_path} have wrong shape: {keypoints.shape}")
                        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                else:
                    logger.warning(f"Keypoints file not found: {keypoints_path}")
                    keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                    
            except Exception as e:
                logger.error(f"Error loading keypoints from {keypoints_path}: {str(e)}")
                keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                
            keypoints_list.append(keypoints)
            
        # Pad list if needed
        if len(keypoints_list) < self.num_frames:
            # Pad with zeros
            padding = [np.zeros((self.num_keypoints, 3), dtype=np.float32)] * (self.num_frames - len(keypoints_list))
            keypoints_list.extend(padding)
            
        return keypoints_list
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filtered_data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing frames and conditions
        """
        try:
            item = self.filtered_data[idx]
            
            # Determine data paths
            data_dir = self.data_dir / item['relative_path']
            video_name = item.get('video_name', '')
            
            # Find the actual video file if video_name is provided
            if video_name:
                video_path = data_dir / video_name
            else:
                # Try to find a video file
                video_files = list(data_dir.glob('*.mp4'))
                if not video_files:
                    logger.warning(f"No video file found in {data_dir}")
                    return self._get_empty_sample()
                video_path = video_files[0]
                
            # Select a random starting frame if total_frames is provided
            start_frame = 0
            if 'total_frames' in item and item['total_frames'] > self.num_frames:
                max_start = item['total_frames'] - self.num_frames
                start_frame = random.randint(0, max_start)
            
            # Load frames
            frames = self._load_frames(data_dir, start_frame)
            
            # Load conditions based on availability
            conditions = {}
            
            # Always load reference image (first frame)
            reference = frames[0].unsqueeze(0)  # Use first frame as reference
            
            # Load audio if available
            if item.get('has_audio', False) and self.condition_ratios.get('audio', 0) > 0:
                try:
                    conditions['audio'] = self._load_audio(video_path)
                except Exception as e:
                    logger.error(f"Error loading audio for {video_path}: {str(e)}")
                    # Add empty audio features
                    conditions['audio'] = torch.zeros((self.num_frames, self.audio_features_dim))
            
            # Load pose if available
            if item.get('has_pose', False) and self.condition_ratios.get('pose', 0) > 0:
                try:
                    # Get frame indices
                    frame_indices = list(range(start_frame, start_frame + self.num_frames))
                    
                    # Load keypoints from cache
                    keypoints_list = self._load_keypoints_from_cache(str(video_path), frame_indices)
                    
                    # If no cached keypoints and keypoint processor available, extract them
                    if all(np.all(k == 0) for k in keypoints_list) and self.keypoint_processor:
                        logger.info(f"No cached keypoints for {video_path}, extracting on-the-fly")
                        # Extract keypoints for each frame
                        new_keypoints = []
                        for i, frame in enumerate(frames):
                            # Convert tensor to numpy
                            frame_np = frame.permute(1, 2, 0).cpu().numpy()
                            # Scale to 0-255 range
                            if frame_np.max() <= 1.0:
                                frame_np = (frame_np * 255).astype(np.uint8)
                            # Extract keypoints
                            keypoints = self._extract_keypoints(frame_np)
                            if keypoints is not None:
                                new_keypoints.append(keypoints)
                                # Cache keypoints
                                keypoints_path = self._get_keypoints_path(str(video_path), start_frame + i)
                                keypoints_path.parent.mkdir(exist_ok=True)
                                np.save(keypoints_path, keypoints)
                            else:
                                new_keypoints.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))
                        
                        # Update keypoints list if we have any valid keypoints
                        if any(not np.all(k == 0) for k in new_keypoints):
                            keypoints_list = new_keypoints
                    
                    # Generate heatmaps
                    pose_heatmaps = self._generate_heatmaps(keypoints_list)
                    conditions['pose'] = pose_heatmaps
                    
                except Exception as e:
                    logger.error(f"Error processing pose for {video_path}: {str(e)}")
                    conditions['pose'] = torch.zeros((self.num_frames, self.num_keypoints, 
                                                   self.heatmap_size[0], self.heatmap_size[1]))
                
            # Load text if available
            if item.get('has_text', False) and self.condition_ratios.get('text', 0) > 0:
                text_embedding_path = data_dir / 'text_embedding.pt'
                conditions['text'] = self._load_text_embedding(text_embedding_path)
                
            # Load reference if available (otherwise use first frame)
            if item.get('has_reference', False):
                reference_path = data_dir / 'reference.jpg'
                if reference_path.exists():
                    reference_img = Image.open(reference_path).convert('RGB')
                    reference = self.reference_transform(reference_img).unsqueeze(0)
                    
            conditions['reference'] = reference
                
            return {
                'frames': frames.permute(1, 0, 2, 3),  # [C, T, H, W] format for 3D models
                'conditions': conditions,
                'metadata': {
                    'id': item.get('id', str(idx)),
                    'duration': item.get('duration', self.num_frames / 30.0),
                    'fps': item.get('fps', 30.0),
                    'video_path': str(video_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            # Track the error
            if 'video_path' in locals():
                self.tracker.dispatch(VideoEventData(
                    video_path=str(video_path),
                    event_type=VideoEvent.PROCESSING_ERROR,
                    details={
                        "error": f"Sample processing error: {str(e)}",
                        "sample_idx": idx,
                        "traceback": traceback.format_exc()
                    }
                ))
            return self._get_empty_sample()
    
    def _get_empty_sample(self) -> Dict[str, Any]:
        """Get an empty sample with zeros."""
        return {
            'frames': torch.zeros((3, self.num_frames, *self.frame_size)),  # [C, T, H, W]
            'conditions': {
                'reference': torch.zeros((1, 3, *self.frame_size))
            },
            'metadata': {
                'id': "empty",
                'duration': self.num_frames / 30.0,
                'fps': 30.0,
                'video_path': ""
            }
        }

        
        
def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    config = OmegaConf.load(config_path)
    return config

def main():

    
    # Load configuration
    config = load_config('omni_config.yaml')
    

    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === SANITY TEST FOR DATASET BEFORE MODEL CREATION ===
    logger.info("=== Running dataset sanity test ===")
    try:
        # Use first stage's condition ratios for testing
        if len(config.get('stages', [])) > 0:
            test_stage = config.stages[0]
            condition_ratios = test_stage.get('condition_ratios', {})
            logger.info(f"Testing with condition ratios: {condition_ratios}")
        else:
            logger.warning("No stages found in config, using empty condition ratios for test")
            condition_ratios = {}
        
        # Check data directory existence
        data_dir = Path(config.data.data_dir)
        logger.info(f"Data directory path: {data_dir}")
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        else:
            logger.info(f"Data directory exists: {data_dir}")
            
            # List contents of data directory
            contents = list(data_dir.glob("*"))
            logger.info(f"Data directory contents: {[str(p.name) for p in contents]}")
            
            # Check for annotation file
            annotation_file = data_dir / 'annotations.json'
            if not annotation_file.exists():
                logger.warning(f"Annotation file not found: {annotation_file}")
                # Create a minimal annotations file for testing
                logger.info("Creating a test annotation file with one sample")
                
                # Find any subdirectories that might contain frames
                subdirs = [p for p in contents if p.is_dir()]
                
                if subdirs:
                    # Create a minimal annotation with the first subdirectory
                    test_annotation = [{
                        'id': '0',
                        'relative_path': subdirs[0].relative_to(data_dir),
                        'duration': config.num_frames / 30.0,
                        'fps': 30.0,
                        'quality_score': 1.0,
                        'motion_score': 1.0,
                        'has_audio': False,
                        'has_pose': False,
                        'has_text': False,
                        'has_reference': True
                    }]
                    
                    # Write test annotation
                    with open(annotation_file, 'w') as f:
                        json.dump(test_annotation, f)
                    logger.info(f"Created test annotation file: {annotation_file}")
                else:
                    logger.error("No subdirectories found in data directory to create test annotation")
        
        # Create a test dataset instance
        logger.info("Instantiating dataset for testing...")
        test_dataset = OmniHumanDataset(
            config=config,
            condition_ratios=condition_ratios
        )
        
        # Check dataset size
        dataset_size = len(test_dataset)
        logger.info(f"Dataset contains {dataset_size} samples")
        
        if dataset_size == 0:
            logger.error("Dataset is empty! Cannot proceed with training.")
            raise ValueError("Dataset is empty! Check your data directory and annotations file.")
        else:
            # Try to load the first item
            logger.info("Attempting to load first item from dataset...")
            first_item = test_dataset[0]
            
            # Print item structure
            logger.info("First item keys: " + str(list(first_item.keys())))
            
            # Print frames shape
            frames = first_item['frames']
            logger.info(f"Frames shape: {frames.shape}")
            
            # Print conditions
            conditions = first_item['conditions']
            logger.info(f"Available conditions: {list(conditions.keys())}")
            
            # Try creating a small dataloader
            logger.info("Testing DataLoader creation...")
            test_loader = DataLoader(
                test_dataset,
                batch_size=min(2, dataset_size),
                shuffle=False,
                num_workers=0  # Use single process for testing
            )
            
            # Try to get one batch
            logger.info("Testing batch loading...")
            test_batch = next(iter(test_loader))
            logger.info(f"Successfully loaded a batch with keys: {list(test_batch.keys())}")
            logger.info(f"Batch frames shape: {test_batch['frames'].shape}")
            
            logger.info("Dataset sanity test PASSED ✅")
    except Exception as e:
        logger.error(f"Dataset sanity test FAILED: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Continue with normal initialization if we haven't exited
    logger.info("=== Continuing normal initialization ===")
    
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