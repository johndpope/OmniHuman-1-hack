from logger import logger
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union




# https://raw.githubusercontent.com/johndpope/EAI/refs/heads/main/pose_vis.py

class SapiensKeypointProcessor:
    """Extracts 308 keypoints using Sapiens model."""
    
    def __init__(
        self,
        checkpoints_dir: str,
        model_name: str = "1b",
        detection_config: Optional[str] = None,
        detection_checkpoint: Optional[str] = None,
        device: str = "cuda",
        heatmap_size: Tuple[int, int] = (64, 64),
    ):
        """Initialize the Sapiens keypoint processor.
        
        Args:
            checkpoints_dir: Directory containing model checkpoints
            model_name: Model size (e.g., "1b", "2b")
            detection_config: Path to detection config file
            detection_checkpoint: Path to detection checkpoint file
            device: Device to run inference on
            heatmap_size: Size of output heatmaps
        """
        self.device = device
        self.heatmap_size = heatmap_size
        self.num_keypoints = 308  # Sapiens Goliath model has 308 keypoints
        
        # Set up transform
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.5/255, 116.5/255, 103.5/255], 
                std=[58.5/255, 57.0/255, 57.5/255]
            )
        ])
        
        # Load Sapiens model
        checkpoint_path = os.path.join(
            checkpoints_dir, 
            f"sapiens_{model_name}/sapiens_{model_name}_goliath_best_goliath_AP_640_torchscript.pt2"
        )
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
        try:
            self.model = torch.jit.load(checkpoint_path)
            self.model.eval()
            self.model.to(device)
            logger.info(f"Loaded Sapiens model from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        # Initialize detector if config and checkpoint are provided
        self.detector = None
        if detection_config and detection_checkpoint:
            try:
                from detector_utils import init_detector, adapt_mmdet_pipeline
                self.detector = init_detector(detection_config, detection_checkpoint, device=device)
                self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
                logger.info(f"Loaded detector from {detection_checkpoint}")
            except ImportError:
                logger.warning("detector_utils not found, detection will not be available")
            except Exception as e:
                logger.error(f"Failed to load detector: {e}")
                
    @torch.inference_mode()
    def extract_keypoints(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extract keypoints from an image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Array of keypoints [num_keypoints, 3] (x, y, confidence)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            
        # Process through detector if available
        if self.detector:
            try:
                from detector_utils import process_images_detector
                image_np = np.array(image)
                image_np = np.expand_dims(image_np, axis=0)
                bboxes_batch = process_images_detector(image_np, self.detector)
                bboxes = self._get_person_bboxes(bboxes_batch[0])
                
                if not bboxes:
                    logger.warning("No person detected in the image")
                    return np.zeros((self.num_keypoints, 3), dtype=np.float32)
                    
                # Use the first person bbox
                bbox = bboxes[0]
                x1, y1, x2, y2 = map(int, bbox[:4])
                cropped_image = image.crop((x1, y1, x2, y2))
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                cropped_image = image  # Fallback to full image
        else:
            cropped_image = image
            
        # Process image through Sapiens model
        input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)
        heatmaps = self.model(input_tensor)
        
        # Convert heatmaps to keypoints
        keypoints = self._heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
        
        return keypoints
    
    def generate_heatmaps(self, keypoints: np.ndarray) -> np.ndarray:
        """Generate Gaussian heatmaps from keypoints.
        
        Args:
            keypoints: Array of keypoints [num_keypoints, 3] (x, y, confidence)
            
        Returns:
            Array of heatmaps [num_keypoints, H, W]
        """
        h, w = self.heatmap_size
        heatmaps = np.zeros((self.num_keypoints, h, w), dtype=np.float32)
        
        for k in range(self.num_keypoints):
            if keypoints[k, 2] > 0.1:  # Confidence threshold
                x, y = keypoints[k, :2]
                
                # Skip if outside the valid range
                if not (0 <= x < 1 and 0 <= y < 1):
                    continue
                    
                # Scale to heatmap size
                x_scaled = int(x * w)
                y_scaled = int(y * h)
                
                # Create 2D Gaussian
                sigma = 2.0
                grid_y = np.arange(h)
                grid_x = np.arange(w)
                grid_y, grid_x = np.meshgrid(grid_y, grid_x, indexing="ij")
                
                dist_squared = (grid_x - x_scaled) ** 2 + (grid_y - y_scaled) ** 2
                exponent = -dist_squared / (2 * sigma ** 2)
                heatmap = np.exp(exponent)
                heatmaps[k] = heatmap
                
        return heatmaps
    
    def _get_person_bboxes(self, bboxes_batch, score_thr=0.3):
        """Extract person bounding boxes from detector output.
        
        Args:
            bboxes_batch: Batch of bounding boxes
            score_thr: Score threshold
            
        Returns:
            List of person bounding boxes
        """
        person_bboxes = []
        for bbox in bboxes_batch:
            if len(bbox) == 5 and bbox[4] > score_thr:
                person_bboxes.append(bbox)
            elif len(bbox) == 4:
                person_bboxes.append(bbox + [1.0])
        return person_bboxes
    
    def _heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> np.ndarray:
        """Convert heatmaps to keypoints.
        
        Args:
            heatmaps: Heatmaps from model [num_keypoints, H, W]
            
        Returns:
            Array of keypoints [num_keypoints, 3] (x, y, confidence)
        """
        num_joints = heatmaps.shape[0]
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        
        for i in range(min(num_joints, self.num_keypoints)):
            heatmap = heatmaps[i]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            conf = heatmap[y, x]
            
            # Normalize coordinates to [0, 1]
            h, w = heatmap.shape
            x_norm = x / w
            y_norm = y / h
            
            keypoints[i] = [x_norm, y_norm, conf]
            
        return keypoints