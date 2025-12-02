"""
YOLO-based player detection for basketball court tracking.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PlayerDetector:
    """Detects players using YOLO object detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize YOLO player detector.

        Args:
            model_name: YOLO model to use (e.g., 'yolov8n.pt', 'yolov8m.pt')
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_name = model_name

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
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_players(
        self,
        frame: np.ndarray,
        filter_court: bool = False,
        court_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict]:
        """
        Detect players in a frame using YOLO.

        Args:
            frame: Input frame (BGR format)
            filter_court: Whether to filter detections to court area
            court_bbox: Court bounding box (x1, y1, x2, y2) for filtering

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box
                - confidence: Detection confidence score
                - center: (cx, cy) center point
                - bottom: (bx, by) bottom center point
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

            logger.debug(f"Detected {len(detections)} players")
            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 255),  # Yellow in BGR
        thickness: int = 3,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            color: BGR color for bounding boxes
            thickness: Line thickness
            show_confidence: Whether to show confidence scores

        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Draw bounding box
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                thickness,
                cv2.LINE_AA
            )

            # Draw confidence score
            if show_confidence:
                text = f"{confidence:.2f}"
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
