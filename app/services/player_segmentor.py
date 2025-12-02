"""
SAM3-based player segmentation for basketball court tracking.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PlayerSegmentor:
    """Segments players using SAM3 (Segment Anything Model 3)."""

    def __init__(
        self,
        model_type: str = "sam2.1_hiera_small",
        device: str = "cpu"
    ):
        """
        Initialize SAM3 player segmentor.

        Args:
            model_type: SAM model to use
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
        """
        self.device = device
        self.model = None
        self.model_type = model_type

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch

            logger.info(f"Loading SAM2 model: {model_type}")

            # Map device for SAM2
            if device == "mps":
                sam_device = "cpu"  # SAM2 may not support MPS directly
                logger.warning("SAM2 may not support MPS, falling back to CPU")
            else:
                sam_device = device

            # Initialize SAM2 predictor
            self.predictor = SAM2ImagePredictor.from_pretrained(
                f"facebook/{model_type}",
                device=sam_device
            )

            logger.info(f"SAM2 model loaded successfully on {sam_device}")

        except ImportError:
            logger.error("sam2 package not installed. Install with: pip install 'git+https://github.com/facebookresearch/sam2.git'")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise

    def segment_players(
        self,
        frame: np.ndarray,
        player_detections: List[Dict]
    ) -> List[Dict]:
        """
        Segment players in a frame using SAM2 with YOLO boxes as prompts.

        Args:
            frame: Input frame (BGR format)
            player_detections: List of YOLO detection dictionaries with 'bbox' keys

        Returns:
            List of segmentation dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box (from YOLO)
                - mask: Binary segmentation mask (H x W)
                - confidence: Detection confidence (from YOLO)
                - center: (cx, cy) center point
                - bottom: (bx, by) bottom center point
        """
        if self.predictor is None or not player_detections:
            logger.debug("No predictor or no detections")
            return []

        try:
            # Convert BGR to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Set image for predictor
            self.predictor.set_image(rgb_frame)

            segmentations = []

            # Process each YOLO detection
            for detection in player_detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox

                # Use bbox as prompt for SAM
                input_box = np.array([x1, y1, x2, y2])

                try:
                    # Predict mask using box prompt
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False
                    )

                    # Get the best mask (first one when multimask_output=False)
                    mask = masks[0]  # Shape: (H, W)
                    score = float(scores[0])

                    segmentations.append({
                        'bbox': bbox,
                        'mask': mask.astype(np.uint8),
                        'confidence': detection['confidence'],
                        'sam_score': score,
                        'center': detection['center'],
                        'bottom': detection['bottom']
                    })

                except Exception as e:
                    logger.debug(f"Failed to segment detection: {e}")
                    continue

            logger.debug(f"Segmented {len(segmentations)} players")
            return segmentations

        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return []

    def draw_segmentations(
        self,
        frame: np.ndarray,
        segmentations: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 255),  # Yellow in BGR
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Draw segmentation masks on frame.

        Args:
            frame: Input frame
            segmentations: List of segmentation dictionaries
            color: BGR color for masks
            alpha: Transparency for mask overlay

        Returns:
            Frame with drawn segmentations
        """
        overlay = frame.copy()

        for segmentation in segmentations:
            mask = segmentation['mask']

            # Apply colored mask
            overlay[mask > 0] = color

        # Blend with original frame
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        return frame

    def check_overlap_with_mask(
        self,
        tag_position: Tuple[int, int],
        segmentations: List[Dict]
    ) -> Optional[int]:
        """
        Check if a tag position overlaps with any segmentation mask.

        Args:
            tag_position: (x, y) tag position in pixels
            segmentations: List of segmentation dictionaries with 'mask' keys

        Returns:
            Index of overlapping segmentation, or None if no overlap
        """
        tx, ty = tag_position

        # Ensure coordinates are within bounds
        if tx < 0 or ty < 0:
            return None

        for idx, segmentation in enumerate(segmentations):
            mask = segmentation['mask']

            # Check bounds
            if ty >= mask.shape[0] or tx >= mask.shape[1]:
                continue

            # Check if tag point is inside mask
            if mask[ty, tx] > 0:
                return idx

        return None
