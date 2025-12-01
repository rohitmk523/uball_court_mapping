"""YOLOv11 detector service for player detection."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.config import DETECTIONS_CACHE_DIR, VIDEO_FILE
from app.core.models import Detection


class PlayerDetector:
    """YOLOv11-based player detector."""

    def __init__(self, model_name: str = "yolo11n.pt", conf_threshold: float = 0.5):
        """
        Initialize player detector.

        Args:
            model_name: YOLO model name (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
            conf_threshold: Confidence threshold for detections
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.person_class_id = 0  # COCO class ID for person

    def load_model(self):
        """Load YOLO model (lazy loading)."""
        if self.model is None:
            print(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            print("YOLO model loaded successfully")

    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players in a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects for detected players
        """
        self.load_model()

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, classes=[self.person_class_id], verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Calculate bottom center of bbox
                bbox_bottom_center_x = (x1 + x2) / 2
                bbox_bottom_center_y = y2  # Bottom of bbox

                detection = Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=conf,
                    class_id=class_id,
                    bbox_bottom_center=(float(bbox_bottom_center_x), float(bbox_bottom_center_y))
                )
                detections.append(detection)

        return detections

    def process_video_batch(
        self,
        video_path: Path = VIDEO_FILE,
        start_frame: int = 0,
        end_frame: int = None,
        frame_skip: int = 1,
        cache: bool = True
    ) -> Dict[int, List[Detection]]:
        """
        Process video frames in batch.

        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (None = end of video)
            frame_skip: Process every Nth frame (1 = every frame)
            cache: Whether to cache results

        Returns:
            Dictionary mapping frame_id to list of detections
        """
        self.load_model()

        # Check cache
        cache_file = DETECTIONS_CACHE_DIR / f"detections_{start_frame}_{end_frame}_{frame_skip}.json"
        if cache and cache_file.exists():
            print(f"Loading cached detections from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Convert back to Detection objects
                result = {}
                for frame_id_str, det_list in cached_data.items():
                    result[int(frame_id_str)] = [Detection(**d) for d in det_list]
                return result

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames

        print(f"Processing video: frames {start_frame} to {end_frame} (skip={frame_skip})")

        detections_per_frame = {}

        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_id = start_frame
        processed_count = 0

        while frame_id < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame if it's in the skip pattern
            if (frame_id - start_frame) % frame_skip == 0:
                detections = self.detect_players(frame)
                detections_per_frame[frame_id] = detections
                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames... (frame {frame_id}/{end_frame})")

            frame_id += 1

        cap.release()

        print(f"Detection complete: {processed_count} frames processed")

        # Cache results
        if cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Convert Detection objects to dicts for JSON serialization
            cache_data = {
                str(frame_id): [det.model_dump() for det in dets]
                for frame_id, dets in detections_per_frame.items()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            print(f"Cached detections to {cache_file}")

        return detections_per_frame

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        draw_bottom_center: bool = True
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input frame
            detections: List of detections
            draw_bottom_center: Whether to draw bottom center point

        Returns:
            Frame with drawn detections
        """
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            label = f"Player {det.confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw bottom center
            if draw_bottom_center:
                cx, cy = det.bbox_bottom_center
                cv2.circle(output, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        return output


# Global detector instance
_detector = None


def get_detector(model_name: str = "yolo11n.pt", conf_threshold: float = 0.5) -> PlayerDetector:
    """
    Get global detector instance (singleton pattern).

    Args:
        model_name: YOLO model name
        conf_threshold: Confidence threshold

    Returns:
        PlayerDetector instance
    """
    global _detector
    if _detector is None:
        _detector = PlayerDetector(model_name, conf_threshold)
    return _detector
