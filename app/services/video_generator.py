"""
Video generation pipeline for creating output videos with UWB tag overlays.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

from .video_processor import VideoProcessor
from .calibration_integration import CalibrationIntegration
from .tag_overlay import TagOverlay
from .player_detector import PlayerDetector

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Generates videos with UWB tag overlays and YOLO player detection."""

    def __init__(
        self,
        video_processor: VideoProcessor,
        calibration: CalibrationIntegration,
        tag_overlay: TagOverlay,
        sync_offset: float = 0.0,
        player_detector: Optional[PlayerDetector] = None,
        use_yolo: bool = True
    ):
        """
        Initialize video generator.

        Args:
            video_processor: Video processing instance
            calibration: Calibration integration instance
            tag_overlay: Tag overlay instance
            sync_offset: Time offset between video and UWB logs (seconds)
            player_detector: YOLO player detector instance (optional)
            use_yolo: Whether to use YOLO detection
        """
        self.video_processor = video_processor
        self.calibration = calibration
        self.tag_overlay = tag_overlay
        self.sync_offset = sync_offset
        self.use_yolo = use_yolo

        # Initialize player detector if not provided
        if use_yolo and player_detector is None:
            try:
                self.player_detector = PlayerDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO detector: {e}")
                self.player_detector = None
                self.use_yolo = False
        else:
            self.player_detector = player_detector

        # Tag data
        self.tag_data: Dict[str, Dict] = {}

    def load_tag_data(self, data_dir: str):
        """
        Load UWB tag data from directory.

        Args:
            data_dir: Directory containing tag JSON files
        """
        logger.info(f"Loading tag data from {data_dir}")
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load all tag JSON files
        tag_files = list(data_path.glob("*.json"))
        logger.info(f"Found {len(tag_files)} tag files")

        for tag_file in tag_files:
            tag_id = tag_file.stem
            try:
                import json
                with open(tag_file, 'r') as f:
                    self.tag_data[tag_id] = json.load(f)
                logger.debug(f"Loaded tag {tag_id}")
            except Exception as e:
                logger.error(f"Failed to load {tag_file}: {e}")

        logger.info(f"Loaded data for {len(self.tag_data)} tags")

    def get_tags_at_frame(
        self,
        frame_number: int,
        time_seconds: float,
        image_width: int,
        image_height: int
    ) -> List[Dict]:
        """
        Get all tag positions for a specific frame using UTC datetime matching.

        Args:
            frame_number: Frame number
            time_seconds: Time in seconds from video start
            image_width: Image width for bounds checking
            image_height: Image height for bounds checking

        Returns:
            List of tag dictionaries with pixel coordinates
        """
        from datetime import datetime, timedelta, timezone

        # Calculate target UTC time for this video frame
        # sync_offset is how many seconds into UWB log the video starts
        target_time = time_seconds + self.sync_offset

        visible_tags = []

        for tag_id, data in self.tag_data.items():
            positions = data.get('positions', [])
            if not positions:
                continue

            # Find closest position by datetime
            closest_pos = None
            min_diff_seconds = float('inf')

            # Get reference datetime from first position to calculate relative times
            try:
                first_datetime_str = positions[0]['datetime']
                # Parse: "2025-11-22 02:26:40.578"
                ref_datetime = datetime.strptime(first_datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                ref_datetime = ref_datetime.replace(tzinfo=timezone.utc)
            except Exception as e:
                logger.debug(f"Failed to parse datetime for tag {tag_id}: {e}")
                continue

            # Check each position
            for pos in positions:
                try:
                    pos_datetime_str = pos['datetime']
                    pos_datetime = datetime.strptime(pos_datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                    pos_datetime = pos_datetime.replace(tzinfo=timezone.utc)

                    # Calculate seconds from ref_datetime
                    pos_time = (pos_datetime - ref_datetime).total_seconds()

                    # Compare with target time
                    diff = abs(pos_time - target_time)

                    if diff < min_diff_seconds:
                        min_diff_seconds = diff
                        closest_pos = pos
                except Exception as e:
                    continue

            # Only include if within 0.5 second tolerance
            if closest_pos and min_diff_seconds < 0.5:
                try:
                    # Project UWB coordinates to image (with rotation)
                    pixel_x, pixel_y = self.calibration.uwb_to_image(
                        closest_pos['x'],
                        closest_pos['y']
                    )

                    # Check if within image bounds
                    if self.calibration.is_point_in_image(
                        pixel_x, pixel_y, image_width, image_height
                    ):
                        visible_tags.append({
                            'tag_id': tag_id,
                            'pixel_x': pixel_x,
                            'pixel_y': pixel_y,
                            'court_x': closest_pos['x'],
                            'court_y': closest_pos['y'],
                            'datetime': closest_pos['datetime']
                        })
                except Exception as e:
                    logger.debug(f"Failed to project tag {tag_id}: {e}")

        return visible_tags

    def process_video(
        self,
        output_path: str,
        duration_seconds: Optional[float] = None,
        start_time: float = 0.0,
        show_trails: bool = True,
        show_info: bool = True,
        codec: str = 'mp4v'
    ) -> bool:
        """
        Process video and generate output with tag overlays.

        Args:
            output_path: Path for output video
            duration_seconds: Duration to process (None = full video)
            start_time: Start time in seconds
            show_trails: Show tag movement trails
            show_info: Show info panel
            codec: Video codec (default: 'mp4v')

        Returns:
            True if successful
        """
        try:
            # Reset tag history
            self.tag_overlay.reset_history()

            # Determine duration
            if duration_seconds is None:
                duration_seconds = self.video_processor.duration - start_time

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            fps = self.video_processor.fps
            video_width = self.video_processor.width
            video_height = self.video_processor.height

            # Calibration was done on 4K (3840x2160)
            # Calculate scale factor from calibration space to video space
            calibration_width = 3840
            calibration_height = 2160
            scale_factor = video_height / calibration_height

            if scale_factor != 1.0:
                logger.info(f"Scaling coordinates from calibration ({calibration_width}x{calibration_height}) to video ({video_width}x{video_height})")
                logger.info(f"Scale factor: {scale_factor:.3f}")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (video_width, video_height)
            )

            if not out.isOpened():
                raise ValueError(f"Failed to create output video: {output_path}")

            logger.info(f"Processing video: {duration_seconds}s from {start_time}s")
            logger.info(f"Output: {output_path}")

            # Process frames
            frame_count = 0
            total_frames = int(duration_seconds * fps)

            for frame_num, time_sec, frame in tqdm(
                self.video_processor.process_duration(duration_seconds, start_time),
                total=total_frames,
                desc="Processing frames"
            ):
                # Detect players with YOLO if enabled
                player_detections = []
                if self.use_yolo and self.player_detector:
                    player_detections = self.player_detector.detect_players(frame)
                    # Draw yellow bounding boxes for players
                    frame = self.player_detector.draw_detections(
                        frame,
                        player_detections,
                        color=(0, 255, 255),  # Yellow in BGR
                        thickness=3
                    )

                # Get UWB tags for this frame (in calibration resolution: 3840x2160)
                tags = self.get_tags_at_frame(frame_num, time_sec, calibration_width, calibration_height)

                # Scale tag coordinates from calibration space to video space
                if scale_factor != 1.0:
                    for tag in tags:
                        tag['pixel_x'] = int(tag['pixel_x'] * scale_factor)
                        tag['pixel_y'] = int(tag['pixel_y'] * scale_factor)

                # Check for overlap and draw tags with appropriate color
                for tag in tags:
                    tag_pos = (tag['pixel_x'], tag['pixel_y'])
                    overlap_idx = None

                    if self.use_yolo and self.player_detector and player_detections:
                        overlap_idx = self.player_detector.check_overlap(
                            tag_pos,
                            player_detections,
                            expansion=int(20 * scale_factor)
                        )

                    # Set color: green if overlap, blue if not
                    if overlap_idx is not None:
                        tag['color'] = (0, 255, 0)  # Green
                    else:
                        tag['color'] = (255, 100, 0)  # Blue

                # Draw tags on frame
                frame = self.tag_overlay.draw_tags(frame, tags, show_trails)

                # Draw info panel if enabled
                if show_info:
                    num_players = len(player_detections) if player_detections else 0
                    frame = self.tag_overlay.draw_info_panel(
                        frame,
                        frame_num,
                        time_sec,
                        len(tags),
                        num_players
                    )

                # Write frame
                out.write(frame)
                frame_count += 1

            # Release resources
            out.release()

            logger.info(f"Successfully processed {frame_count} frames")
            logger.info(f"Output saved to: {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            return False

    def generate_preview_frame(
        self,
        frame_number: int,
        show_trails: bool = False
    ) -> Optional[np.ndarray]:
        """
        Generate a single preview frame with tags.

        Args:
            frame_number: Frame number to preview
            show_trails: Show tag trails

        Returns:
            Frame with tags or None if failed
        """
        try:
            frame = self.video_processor.get_frame(frame_number)
            if frame is None:
                return None

            time_sec = frame_number / self.video_processor.fps
            width = self.video_processor.width
            height = self.video_processor.height

            tags = self.get_tags_at_frame(frame_number, time_sec, width, height)
            frame = self.tag_overlay.draw_tags(frame, tags, show_trails)
            frame = self.tag_overlay.draw_info_panel(
                frame,
                frame_number,
                time_sec,
                len(tags)
            )

            return frame

        except Exception as e:
            logger.error(f"Failed to generate preview: {e}")
            return None
