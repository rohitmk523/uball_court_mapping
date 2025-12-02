"""
Video processing service for UWB tag overlay on GoPro footage.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video loading, frame extraction, and basic processing."""

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps:.2f}")
        logger.info(f"  Total frames: {self.total_frames}")
        logger.info(f"  Duration: {self.duration:.2f}s")

    def get_properties(self) -> Dict[str, Any]:
        """Get video properties."""
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": self.duration
        }

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video.

        Args:
            frame_number: Frame index (0-based)

        Returns:
            Frame as numpy array (BGR) or None if failed
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            logger.warning(f"Frame number {frame_number} out of range")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            logger.warning(f"Failed to read frame {frame_number}")
            return None

        return frame

    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at specific time.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame as numpy array (BGR) or None if failed
        """
        frame_number = int(time_seconds * self.fps)
        return self.get_frame(frame_number)

    def frame_generator(self, start_frame: int = 0, end_frame: Optional[int] = None):
        """
        Generator that yields frames sequentially.

        Args:
            start_frame: Starting frame number (default: 0)
            end_frame: Ending frame number (default: total_frames)

        Yields:
            Tuple of (frame_number, frame)
        """
        if end_frame is None:
            end_frame = self.total_frames

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_num}, stopping")
                break

            yield frame_num, frame

    def process_duration(self, duration_seconds: float, start_time: float = 0.0):
        """
        Generator for processing a specific duration of video.

        Args:
            duration_seconds: Duration to process in seconds
            start_time: Start time in seconds (default: 0)

        Yields:
            Tuple of (frame_number, time_seconds, frame)
        """
        start_frame = int(start_time * self.fps)
        end_frame = int((start_time + duration_seconds) * self.fps)
        end_frame = min(end_frame, self.total_frames)

        logger.info(f"Processing {duration_seconds}s from {start_time}s")
        logger.info(f"  Frames: {start_frame} to {end_frame}")

        for frame_num, frame in self.frame_generator(start_frame, end_frame):
            time_sec = frame_num / self.fps
            yield frame_num, time_sec, frame

    def close(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            logger.info("Video processor closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        self.close()
