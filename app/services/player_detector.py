"""
YOLO-based player detection for basketball court tracking.
Enhanced with optional SAM2 segmentation for improved accuracy.
"""
import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import ByteTrack tracker
try:
    from app.services.bytetrack_tracker import ByteTrackTracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    logger.warning("ByteTrack tracker not available. Tracking functionality disabled.")

# Import SAM2 segmenter
try:
    from app.services.sam2_segmenter import SAM2Segmenter, check_sam2_availability
    SAM2_AVAILABLE = check_sam2_availability()
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning("SAM2 segmenter not available. Mask-based features disabled.")


class PlayerDetector:
    """Detects players using YOLO object detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        enable_tracking: bool = False,
        track_buffer: int = 50,
        sam2_config_path: Optional[str] = None,
        use_sam2: Optional[bool] = None
    ):
        """
        Initialize YOLO player detector with optional SAM2 segmentation.

        Args:
            model_name: YOLO model to use (e.g., 'yolov8n.pt', 'yolov8m.pt')
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
            enable_tracking: Whether to enable ByteTrack tracking
            track_buffer: Number of frames to keep lost tracks
            sam2_config_path: Path to SAM2 config file (default: config/sam2_config.json)
            use_sam2: Override to enable/disable SAM2 (default: use env var USE_SAM2)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_name = model_name
        self.enable_tracking = enable_tracking
        self.tracker = None
        self.segmenter = None

        # SAM2 configuration
        self.use_sam2 = self._should_use_sam2(use_sam2)
        self.sam2_config = self._load_sam2_config(sam2_config_path)

        try:
            from ultralytics import YOLO
            import torch
            from functools import wraps

            # Monkey-patch torch.load to use weights_only=False for YOLO models
            original_load = torch.load

            @wraps(original_load)
            def patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)

            torch.load = patched_load

            logger.info(f"Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)
            self.model.to(device)
            logger.info(f"YOLO model loaded successfully on {device}")

            # Restore original torch.load
            torch.load = original_load

            # Initialize ByteTrack tracker if enabled
            if self.enable_tracking and BYTETRACK_AVAILABLE:
                use_mask_features = self.sam2_config.get('tracking', {}).get('use_mask_features', False)
                self.tracker = ByteTrackTracker(
                    track_activation_threshold=0.25,
                    lost_track_buffer=track_buffer,
                    minimum_matching_threshold=0.8,
                    frame_rate=30,
                    use_mask_features=use_mask_features and self.use_sam2
                )
                logger.info("ByteTrack tracking enabled")
            elif self.enable_tracking and not BYTETRACK_AVAILABLE:
                logger.warning("Tracking requested but ByteTrack not available")

            # Initialize SAM2 segmenter if enabled
            if self.use_sam2 and SAM2_AVAILABLE:
                try:
                    sam2_cfg = self.sam2_config.get('sam2', {})
                    self.segmenter = SAM2Segmenter(
                        model_cfg=sam2_cfg.get('model_cfg', 'sam2_hiera_l.yaml'),
                        checkpoint_path=sam2_cfg.get('checkpoint_path', 'checkpoints/sam2_hiera_large.pt'),
                        device=self.device,
                        min_mask_area=sam2_cfg.get('min_mask_area', 1000),
                        stability_score_thresh=sam2_cfg.get('stability_score_thresh', 0.88),
                        box_expansion=sam2_cfg.get('box_expansion', 10)
                    )
                    logger.info("SAM2 segmentation enabled")
                except Exception as e:
                    logger.error(f"Failed to initialize SAM2: {e}")
                    logger.warning("Continuing without SAM2 segmentation")
                    self.use_sam2 = False
            elif self.use_sam2 and not SAM2_AVAILABLE:
                logger.warning("SAM2 requested but not available. Install from: https://github.com/facebookresearch/segment-anything-2")
                self.use_sam2 = False

        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _should_use_sam2(self, use_sam2: Optional[bool]) -> bool:
        """Determine if SAM2 should be enabled based on env var or override"""
        if use_sam2 is not None:
            return use_sam2

        # Check environment variable
        env_use_sam2 = os.getenv('USE_SAM2', 'false').lower()
        return env_use_sam2 in ('true', '1', 'yes', 'on')

    def _load_sam2_config(self, config_path: Optional[str]) -> Dict:
        """Load SAM2 configuration from file"""
        if config_path is None:
            config_path = "config/sam2_config.json"

        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"SAM2 config not found: {config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded SAM2 config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load SAM2 config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default SAM2 configuration"""
        return {
            'sam2': {
                'enabled': False,
                'model_cfg': 'sam2_hiera_l.yaml',
                'checkpoint_path': 'checkpoints/sam2_hiera_large.pt',
                'device': 'cpu',
                'min_mask_area': 1000,
                'stability_score_thresh': 0.88,
                'box_expansion': 10
            },
            'visualization': {
                'show_masks': True,
                'mask_alpha': 0.5,
                'mask_only': False
            },
            'tracking': {
                'use_mask_features': True,
                'use_mask_centroid': True,
                'use_refined_bbox': True
            }
        }

    def detect_players(
        self,
        frame: np.ndarray,
        filter_court: bool = False,
        court_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict]:
        """
        Detect players in a frame using YOLO, with optional SAM2 segmentation.

        Args:
            frame: Input frame (BGR format)
            filter_court: Whether to filter detections to court area
            court_bbox: Court bounding box (x1, y1, x2, y2) for filtering

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box (refined if SAM2 enabled)
                - confidence: Detection confidence score
                - center: (cx, cy) center point (refined if SAM2 enabled)
                - bottom: (bx, by) bottom center point (refined if SAM2 enabled)
                - mask: Binary mask (H, W) uint8 (if SAM2 enabled)
                - refined_bbox: Tighter bbox from mask (if SAM2 enabled)
                - mask_centroid: Mask centroid (if SAM2 enabled)
                - mask_area: Mask area in pixels (if SAM2 enabled)
                - mask_confidence: SAM2 stability score (if SAM2 enabled)
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=[0],  # 0 = person class in COCO dataset
                verbose=False
            )

            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    # Calculate center and bottom points
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    bx = int((x1 + x2) / 2)
                    by = int(y2)  # Bottom of bounding box

                    # Filter by court bbox if specified
                    if filter_court and court_bbox:
                        cx1, cy1, cx2, cy2 = court_bbox
                        if not (cx1 <= bx <= cx2 and cy1 <= by <= cy2):
                            continue

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'center': (cx, cy),
                        'bottom': (bx, by)
                    })

            logger.debug(f"YOLO detected {len(detections)} players")

            # Apply SAM2 segmentation if enabled
            if self.use_sam2 and self.segmenter is not None and detections:
                detections = self.segmenter.segment_players(frame, detections)
                logger.debug(f"SAM2 enhanced {len(detections)} players")

            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),  # Green in BGR
        thickness: int = 2,
        show_confidence: bool = True,
        show_masks: bool = True,
        color_by_tag_id: bool = False,
        id_mapper = None
    ) -> np.ndarray:
        """
        Draw detection bounding boxes and masks on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            color: BGR color for bounding boxes
            thickness: Line thickness
            show_confidence: Whether to show confidence scores
            show_masks: Whether to show SAM2 masks (if available)

        Returns:
            Frame with drawn detections
        """
        # Draw masks first (if enabled and available)
        if show_masks and self.use_sam2 and self.segmenter is not None:
            mask_alpha = self.sam2_config.get('visualization', {}).get('mask_alpha', 0.5)
            frame = self.segmenter.visualize_masks(
                frame,
                detections,
                alpha=mask_alpha,
                color_by_tag_id=color_by_tag_id,
                id_mapper=id_mapper
            )
            # Don't return - continue to draw bounding boxes and IDs on top

        # Draw bounding boxes, track IDs, and confidence scores
        for detection in detections:
            bbox = detection.get('refined_bbox', detection['bbox'])
            confidence = detection['confidence']
            track_id = detection.get('track_id', None)

            # Convert bbox to integers
            bbox = [int(x) for x in bbox]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                thickness,
                cv2.LINE_AA
            )

            # Prepare label text (track ID + confidence)
            uwb_tag_id = detection.get('uwb_tag_id')

            if id_mapper:
                text = id_mapper.get_label(track_id, uwb_tag_id, mode='dual')
                if show_confidence:
                    text += f" {confidence:.2f}"
            elif track_id is not None and show_confidence:
                text = f"ID:{track_id} {confidence:.2f}"
            elif track_id is not None:
                text = f"ID:{track_id}"
            elif show_confidence:
                text = f"{confidence:.2f}"
            else:
                text = None

            # Draw label if we have one
            if text:
                text_size, _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                # Draw text background
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1] - text_size[1] - 8),
                    (bbox[0] + text_size[0] + 8, bbox[1]),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    frame,
                    text,
                    (bbox[0] + 4, bbox[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

        return frame

    def check_overlap(
        self,
        tag_position: Tuple[int, int],
        detections: List[Dict],
        expansion: int = 10
    ) -> Optional[int]:
        """
        Check if a tag position overlaps with any detection.

        Args:
            tag_position: (x, y) tag position in pixels
            detections: List of detection dictionaries
            expansion: Pixels to expand bbox for overlap check

        Returns:
            Index of overlapping detection, or None if no overlap
        """
        tx, ty = tag_position

        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Expand bbox slightly
            x1 -= expansion
            y1 -= expansion
            x2 += expansion
            y2 += expansion

            # Check if tag point is inside bbox
            if x1 <= tx <= x2 and y1 <= ty <= y2:
                return idx

        return None

    def track_players(
        self,
        frame: np.ndarray,
        filter_court: bool = False,
        court_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict]:
        """
        Detect and track players with persistent IDs.

        Args:
            frame: Input frame (BGR format)
            filter_court: Whether to filter detections to court area
            court_bbox: Court bounding box (x1, y1, x2, y2) for filtering

        Returns:
            List of tracked players with persistent track_id and detection data
        """
        if not self.enable_tracking or self.tracker is None:
            logger.warning("Tracking not enabled. Use detect_players() instead or enable tracking in __init__")
            return self.detect_players(frame, filter_court, court_bbox)

        # First, detect players
        detections = self.detect_players(frame, filter_court, court_bbox)

        if not detections:
            # Even if no detections, update tracker (keeps existing tracks alive)
            return self.tracker.update([], frame)

        # Update tracker with detections to get persistent IDs
        tracked_players = self.tracker.update(detections, frame)

        logger.debug(f"Tracked {len(tracked_players)} players with IDs")
        return tracked_players

    def get_track_statistics(self) -> Optional[Dict]:
        """
        Get tracking statistics if tracking is enabled.

        Returns:
            Dictionary with tracking stats or None if tracking disabled
        """
        if self.tracker is not None:
            return self.tracker.get_track_statistics()
        return None

    def reset_tracker(self):
        """Reset the tracker state (useful for starting new video sequences)"""
        if self.tracker is not None:
            self.tracker.reset()
            logger.info("Tracker reset")
        else:
            logger.warning("No tracker to reset")
