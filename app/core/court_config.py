"""
Configuration for court video generation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class CourtVideoConfig:
    """Configuration for court-centric video generation."""

    # Input paths
    video_path: str
    calibration_file: str = "data/calibration/homography.json"
    court_image_file: str = "data/calibration/court_image.png"
    tags_dir: str = "data/tags"
    log_file: str = "data/logs/session_1763778442483.log"

    # Synchronization
    sync_offset_seconds: Optional[float] = None  # Manual override
    auto_sync: bool = True  # Auto-calculate if None

    # YOLO parameters
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.5
    yolo_device: str = "mps"  # "cpu", "cuda", or "mps" for Mac

    # Visualization
    tagging_threshold_cm: float = 200.0
    uwb_tag_radius_px: int = 18
    uwb_circle_radius_cm: float = 200.0
    player_radius_px: int = 12
    uwb_tag_color: Tuple[int, int, int] = (255, 100, 0)  # Blue in BGR
    player_untagged_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow in BGR
    player_tagged_color: Tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White in BGR
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black in BGR

    # Text rendering
    font_scale: float = 0.8
    font_thickness: int = 2

    # Court canvas
    court_margin_cm: float = 200.0  # Margin around court for tags outside

    # Output
    output_video_path: str = "data/output/court_view_{timestamp}.mp4"
    output_events_path: str = "data/output/tag_events_{timestamp}.json"
    output_fps: float = 29.97
    codec: str = "mp4v"

    # Processing options
    start_frame: int = 0
    end_frame: Optional[int] = None  # None = process until end
    max_frames: Optional[int] = None  # None = no limit

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required files exist
        video_file = Path(self.video_path)
        if not video_file.exists():
            return False, f"Video file not found: {self.video_path}"

        calibration_file = Path(self.calibration_file)
        if not calibration_file.exists():
            return False, f"Calibration file not found: {self.calibration_file}"

        court_image_file = Path(self.court_image_file)
        if not court_image_file.exists():
            return False, f"Court image file not found: {self.court_image_file}"

        tags_dir = Path(self.tags_dir)
        if not tags_dir.exists() or not tags_dir.is_dir():
            return False, f"Tags directory not found: {self.tags_dir}"

        # Check tags directory has JSON files
        tag_files = list(tags_dir.glob("*.json"))
        if not tag_files:
            return False, f"No tag JSON files found in: {self.tags_dir}"

        # Validate numeric parameters
        if self.tagging_threshold_cm <= 0:
            return False, "tagging_threshold_cm must be positive"

        if self.yolo_confidence < 0 or self.yolo_confidence > 1:
            return False, "yolo_confidence must be between 0 and 1"

        if self.output_fps <= 0:
            return False, "output_fps must be positive"

        if self.start_frame < 0:
            return False, "start_frame must be non-negative"

        if self.end_frame is not None and self.end_frame <= self.start_frame:
            return False, "end_frame must be greater than start_frame"

        if self.max_frames is not None and self.max_frames <= 0:
            return False, "max_frames must be positive"

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if self.yolo_device not in valid_devices:
            return False, f"yolo_device must be one of: {valid_devices}"

        return True, None

    def get_output_paths(self, timestamp: str) -> Tuple[str, str]:
        """
        Get output paths with timestamp substituted.

        Args:
            timestamp: Timestamp string to substitute

        Returns:
            Tuple of (video_path, events_path)
        """
        video_path = self.output_video_path.replace("{timestamp}", timestamp)
        events_path = self.output_events_path.replace("{timestamp}", timestamp)
        return video_path, events_path
