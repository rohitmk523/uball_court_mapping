"""SAM2 Segmenter for Basketball Player Tracking

Provides precise player segmentation using Segment Anything Model 2 (SAM2).
Used in conjunction with YOLO detection for enhanced tracking quality.
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# SAM2 dependencies
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning("SAM2 not available. Install from: https://github.com/facebookresearch/segment-anything-2")


class SAM2Segmenter:
    """SAM2-based player segmentation for enhanced tracking"""

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l.yaml",
        checkpoint_path: str = "checkpoints/sam2_hiera_large.pt",
        device: str = "cpu",
        min_mask_area: int = 1000,
        stability_score_thresh: float = 0.88,
        box_expansion: int = 10
    ):
        """
        Initialize SAM2 Segmenter

        Args:
            model_cfg: SAM2 model configuration (yaml file name)
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
            min_mask_area: Minimum mask area (pixels) to keep
            stability_score_thresh: Stability score threshold for masks
            box_expansion: Pixels to expand YOLO bounding box for SAM2 prompting
        """
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 not installed. Install from: "
                "https://github.com/facebookresearch/segment-anything-2"
            )

        self.device = device
        self.min_mask_area = min_mask_area
        self.stability_score_thresh = stability_score_thresh
        self.box_expansion = box_expansion

        # Initialize SAM2
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.predictor = self._load_sam2()

        logger.info(f"SAM2 Segmenter initialized on {device}")
        logger.info(f"  Model: {model_cfg}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Min mask area: {min_mask_area}px")

    def _load_sam2(self) -> SAM2ImagePredictor:
        """Load SAM2 model"""
        try:
            # Suppress verbose SAM2 internal logs
            sam2_logger = logging.getLogger('root')
            original_level = sam2_logger.level
            sam2_logger.setLevel(logging.WARNING)

            # Build SAM2 model
            sam2_model = build_sam2(
                self.model_cfg,
                self.checkpoint_path,
                device=self.device,
                apply_postprocessing=False
            )

            # Create predictor
            predictor = SAM2ImagePredictor(sam2_model)

            # Restore logging level
            sam2_logger.setLevel(original_level)

            logger.info(f"SAM2 model loaded: {self.model_cfg}")
            return predictor

        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise

    def segment_players(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Segment players using SAM2 with YOLO detections as prompts.

        Args:
            frame: Input frame (BGR format)
            detections: List of YOLO detections with 'bbox' key

        Returns:
            Enhanced detections with added mask and refined bbox data:
                - mask: Binary mask (H, W) uint8
                - refined_bbox: Tighter bounding box from mask
                - mask_centroid: (cx, cy) mask centroid
                - mask_area: Mask area in pixels
                - mask_confidence: SAM2 stability score
        """
        if not detections:
            return []

        # Convert frame to RGB for SAM2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set image for SAM2 predictor
        self.predictor.set_image(frame_rgb)

        enhanced_detections = []

        for detection in detections:
            bbox = detection['bbox']

            # Expand bounding box slightly for better segmentation
            expanded_bbox = self._expand_bbox(bbox, frame.shape)

            # Convert bbox to SAM2 format [x1, y1, x2, y2]
            box_prompt = np.array([
                expanded_bbox[0], expanded_bbox[1],
                expanded_bbox[2], expanded_bbox[3]
            ], dtype=np.float32)

            try:
                # Predict mask using box prompt
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_prompt[None, :],
                    multimask_output=False
                )

                # Get best mask
                if len(masks) > 0:
                    mask = masks[0]  # Shape: (H, W) bool
                    score = float(scores[0])

                    # Filter by stability score
                    if score < self.stability_score_thresh:
                        logger.debug(f"Mask rejected (low score: {score:.3f})")
                        continue

                    # Convert to uint8
                    mask_uint8 = (mask * 255).astype(np.uint8)

                    # Calculate mask properties
                    mask_area = int(np.sum(mask))

                    # Filter by area
                    if mask_area < self.min_mask_area:
                        logger.debug(f"Mask rejected (small area: {mask_area}px)")
                        continue

                    # Calculate refined bounding box from mask
                    refined_bbox = self._mask_to_bbox(mask_uint8)

                    # Calculate mask centroid
                    centroid = self._calculate_centroid(mask_uint8)

                    # Create enhanced detection
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'mask': mask_uint8,
                        'refined_bbox': refined_bbox,
                        'mask_centroid': centroid,
                        'mask_area': mask_area,
                        'mask_confidence': score,
                        # Update center and bottom to use refined bbox
                        'center': [
                            (refined_bbox[0] + refined_bbox[2]) / 2,
                            (refined_bbox[1] + refined_bbox[3]) / 2
                        ],
                        'bottom': [
                            (refined_bbox[0] + refined_bbox[2]) / 2,
                            refined_bbox[3]
                        ]
                    })

                    enhanced_detections.append(enhanced_detection)

            except Exception as e:
                logger.warning(f"SAM2 segmentation failed for detection: {e}")
                # Fall back to original detection without mask
                enhanced_detections.append(detection)

        logger.debug(f"SAM2 segmented {len(enhanced_detections)}/{len(detections)} players")
        return enhanced_detections

    def _expand_bbox(
        self,
        bbox: List[float],
        frame_shape: Tuple[int, int, int]
    ) -> List[int]:
        """Expand bounding box by expansion margin"""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        # Expand
        x1 = max(0, int(x1 - self.box_expansion))
        y1 = max(0, int(y1 - self.box_expansion))
        x2 = min(w, int(x2 + self.box_expansion))
        y2 = min(h, int(y2 + self.box_expansion))

        return [x1, y1, x2, y2]

    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Calculate tight bounding box from mask"""
        # Find non-zero pixels
        coords = cv2.findNonZero(mask)

        if coords is None or len(coords) == 0:
            return [0, 0, 0, 0]

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)
        return [x, y, x + w, y + h]

    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Calculate mask centroid using moments"""
        # Calculate moments
        moments = cv2.moments(mask)

        if moments['m00'] == 0:
            return (0.0, 0.0)

        # Calculate centroid
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        return (float(cx), float(cy))

    def visualize_masks(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        alpha: float = 0.5,
        mask_only: bool = False,
        color_by_tag_id: bool = False,
        id_mapper = None
    ) -> np.ndarray:
        """
        Visualize segmentation masks on frame.

        Args:
            frame: Input frame
            detections: List of detections with 'mask' key
            alpha: Mask overlay transparency (0-1)
            mask_only: If True, show only mask overlay (no bbox/text)

        Returns:
            Frame with mask visualization
        """
        output = frame.copy()

        if mask_only:
            # Create black background
            output = np.zeros_like(frame)

        # Generate colors for each detection
        if color_by_tag_id and id_mapper is not None:
            # Use persistent colors from UWB tag_id
            colors = []
            for detection in detections:
                tag_id = detection.get('uwb_tag_id')
                colors.append(id_mapper.get_color_for_tag(tag_id))
        else:
            # Original behavior: color by index
            colors = self._generate_colors(len(detections))

        for i, detection in enumerate(detections):
            if 'mask' not in detection:
                continue

            mask = detection['mask']
            color = colors[i]

            # Create colored mask overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color

            # Blend with frame
            if mask_only:
                output[mask > 0] = color
            else:
                output = cv2.addWeighted(output, 1, colored_mask, alpha, 0)

            # Draw refined bbox and info (if not mask_only)
            if not mask_only and 'refined_bbox' in detection:
                bbox = detection['refined_bbox']

                # Draw bounding box
                cv2.rectangle(
                    output,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color,
                    2,
                    cv2.LINE_AA
                )

                # Draw centroid
                if 'mask_centroid' in detection:
                    cx, cy = detection['mask_centroid']
                    cv2.circle(output, (int(cx), int(cy)), 5, color, -1)

                # Draw track ID if available
                if 'track_id' in detection:
                    track_id = detection['track_id']
                    uwb_tag_id = detection.get('uwb_tag_id')

                    # Show both IDs if mapper provided
                    if id_mapper:
                        label = id_mapper.get_label(track_id, uwb_tag_id, mode='dual')
                    elif uwb_tag_id is not None:
                        label = f"Tag:{uwb_tag_id}"
                    else:
                        label = f"ID:{track_id}"

                    # Draw label background
                    text_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        output,
                        (bbox[0], bbox[1] - text_size[1] - 8),
                        (bbox[0] + text_size[0] + 8, bbox[1]),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        output,
                        label,
                        (bbox[0] + 4, bbox[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        return output

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1))
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        return colors

    def get_mask_features(self, detection: Dict) -> Optional[Dict]:
        """
        Extract useful features from mask for tracking.

        Args:
            detection: Detection with 'mask' key

        Returns:
            Dictionary of mask features or None if no mask
        """
        if 'mask' not in detection:
            return None

        mask = detection['mask']

        # Calculate contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get largest contour (should be the player)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Fit ellipse if enough points
        ellipse = None
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)

        # Calculate hu moments for shape matching
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        return {
            'area': area,
            'perimeter': perimeter,
            'ellipse': ellipse,
            'hu_moments': hu_moments,
            'contour': largest_contour
        }


def check_sam2_availability() -> bool:
    """Check if SAM2 is available"""
    return SAM2_AVAILABLE


def get_default_sam2_config() -> Dict:
    """Get default SAM2 configuration"""
    return {
        'enabled': False,
        'model_cfg': 'sam2_hiera_l.yaml',
        'checkpoint_path': 'checkpoints/sam2_hiera_large.pt',
        'device': 'cpu',
        'min_mask_area': 1000,
        'stability_score_thresh': 0.88,
        'box_expansion': 10
    }
