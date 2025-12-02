"""
Court-centric video generator for player-tag interaction visualization.

This service creates videos showing basketball court as the primary canvas,
with players projected from video onto the court, and UWB tags overlaid.
"""
import cv2
import numpy as np
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from app.core.court_config import CourtVideoConfig
from app.models.court_events import (
    TagEvent, PlayerCourtPosition, UWBTagPosition,
    ProcessingMetadata, ProcessingResult, TagEventLogger
)
from app.services.calibration_integration import CalibrationIntegration
from app.services.player_detector import PlayerDetector
from app.services.synchronizer import Synchronizer
from app.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class CourtVideoGenerator:
    """Generates court-centric videos with player-tag visualization."""

    def __init__(self, config: CourtVideoConfig):
        """
        Initialize court video generator.

        Args:
            config: CourtVideoConfig with all parameters
        """
        self.config = config

        # Validate configuration
        is_valid, error_msg = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_msg}")

        # Initialize services
        self.calibration = CalibrationIntegration(config.calibration_file)
        self.player_detector = PlayerDetector(
            model_name=config.yolo_model,
            confidence_threshold=config.yolo_confidence,
            device=config.yolo_device
        )
        self.synchronizer = Synchronizer()
        self.video_processor = VideoProcessor(config.video_path)

        # Event logger
        self.event_logger = TagEventLogger()

        # Court canvas template
        self.court_canvas_template: Optional[np.ndarray] = None
        self.canvas_height: int = 0
        self.canvas_width: int = 0

        # Scale factors for court coordinates to pixels
        self.scale_x: float = 0.0
        self.scale_y: float = 0.0

        # Sync offset
        self.sync_offset: float = 0.0

        # Tag data cache
        self.tag_data: Dict[int, List[Dict]] = {}

        logger.info("CourtVideoGenerator initialized")

    def load_court_canvas(self) -> np.ndarray:
        """
        Load and rotate court image as canvas.

        Returns:
            Rotated court image (4261 x 7341 after rotation)
        """
        logger.info(f"Loading court image from {self.config.court_image_file}")
        court_img = cv2.imread(self.config.court_image_file)

        if court_img is None:
            raise FileNotFoundError(f"Failed to load court image: {self.config.court_image_file}")

        # Rotate 90° clockwise
        rotated = cv2.rotate(court_img, cv2.ROTATE_90_CLOCKWISE)

        self.canvas_height, self.canvas_width = rotated.shape[:2]
        logger.info(f"Court canvas dimensions: {self.canvas_width} x {self.canvas_height}")

        # Calculate scale factors
        # After rotation: width corresponds to court width (1524 cm)
        #                height corresponds to court length (2865 cm)
        self.scale_x = self.canvas_width / CalibrationIntegration.COURT_WIDTH_CM
        self.scale_y = self.canvas_height / CalibrationIntegration.COURT_LENGTH_CM

        logger.info(f"Scale factors: X={self.scale_x:.3f} px/cm, Y={self.scale_y:.3f} px/cm")

        return rotated

    def load_tag_data(self):
        """Load UWB tag data from JSON files."""
        logger.info(f"Loading tag data from {self.config.tags_dir}")
        tags_dir = Path(self.config.tags_dir)

        for tag_file in tags_dir.glob("*.json"):
            try:
                with open(tag_file, 'r') as f:
                    data = json.load(f)
                    tag_id = data['tag_id']
                    positions = data['positions']
                    self.tag_data[tag_id] = positions
                    logger.debug(f"Loaded {len(positions)} positions for tag {tag_id}")
            except Exception as e:
                logger.error(f"Failed to load tag file {tag_file}: {e}")

        logger.info(f"Loaded data for {len(self.tag_data)} tags")

    def calculate_sync_offset(self):
        """Calculate or use configured sync offset."""
        if self.config.sync_offset_seconds is not None:
            self.sync_offset = self.config.sync_offset_seconds
            logger.info(f"Using manual sync offset: {self.sync_offset} seconds")
        elif self.config.auto_sync:
            self.sync_offset = self.synchronizer.calculate_sync_offset(
                self.config.video_path,
                self.config.log_file
            )
            logger.info(f"Auto-calculated sync offset: {self.sync_offset} seconds")
        else:
            self.sync_offset = 0.0
            logger.info("No sync offset applied")

    # Calibration offsets based on homography.json analysis
    # Offsets to convert from calibration space to UWB space
    # Based on calibration point ranges in homography.json:
    # - Calibration X: 699 to 3552 → UWB X: 0 to 2853
    # - Calibration Y: 1511 to 3672 → UWB Y: 0 to 2161
    CALIB_OFFSET_X = 699.0
    CALIB_OFFSET_Y = 1511.0  # Fixed from 1200 based on actual calibration range

    def uwb_to_canvas_pixels(self, x_uwb: float, y_uwb: float) -> Tuple[int, int]:
        """
        Transform UWB coordinates to canvas pixels.
        
        Canvas is rotated 90° Clockwise relative to UWB/Standard Court.
        UWB: X=Length (0-2865), Y=Width (0-1524)
        Canvas: Vertical. Width=1524, Height=2865.
        
        Rotation 90° CW:
        (x, y) -> (h - y, x)
        where h is the height of the source (which is Width of court = 1524)
        
        So:
        Canvas X = (1524 - y_uwb) * scale_x
        Canvas Y = x_uwb * scale_y
        
        Args:
            x_uwb: X in UWB system (cm)
            y_uwb: Y in UWB system (cm)

        Returns:
            Tuple of (pixel_x, pixel_y) on canvas
        """
        # Canvas X corresponds to inverted UWB Y
        pixel_x = int((CalibrationIntegration.COURT_WIDTH_CM - y_uwb) * self.scale_x)
        
        # Canvas Y corresponds to UWB X
        pixel_y = int(x_uwb * self.scale_y)

        return pixel_x, pixel_y

    def project_player_to_court(self, detection: Dict) -> Optional[PlayerCourtPosition]:
        """
        Project player from video to rotated court canvas.

        The homography was calibrated with pixel coordinates on the rotated court canvas,
        so image_to_court() returns canvas pixel coordinates directly!

        Args:
            detection: YOLO detection dictionary

        Returns:
            PlayerCourtPosition or None if projection fails
        """
        try:
            # Get bottom-center point from detection
            bx, by = detection['bottom']

            # Project using homography - returns CANVAS PIXEL coordinates!
            canvas_x, canvas_y = self.calibration.image_to_court(bx, by)

            # Convert to integers
            canvas_x = int(round(canvas_x))
            canvas_y = int(round(canvas_y))

            # Bounds check - ensure within canvas dimensions
            if not self.is_within_court_bounds(canvas_x, canvas_y):
                return None

            return PlayerCourtPosition(
                detection_index=detection.get('index', 0),
                bbox=detection['bbox'],
                video_bottom_x=bx,
                video_bottom_y=by,
                court_x=canvas_x,  # Actually canvas pixel X
                court_y=canvas_y,  # Actually canvas pixel Y
                canvas_x=canvas_x,
                canvas_y=canvas_y,
                confidence=detection['confidence']
            )

        except Exception as e:
            logger.debug(f"Failed to project player: {e}")
            return None

    def is_within_court_bounds(self, x_cm: float, y_cm: float) -> bool:
        """
        TEMPORARY: Disabled bounds check to see where all players project.

        This allows us to visualize ALL player projections without filtering,
        so we can verify the coordinate transformation is working correctly.

        Will re-enable proper bounds checking after validation.
        """
        # TEMPORARY: Allow all projections through
        return True

    def calculate_distance_cm(
        self,
        player_pos: Tuple[float, float],
        tag_pos: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance in court space (cm).

        Args:
            player_pos: (x, y) in cm
            tag_pos: (x, y) in cm

        Returns:
            Distance in cm
        """
        dx = player_pos[0] - tag_pos[0]
        dy = player_pos[1] - tag_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_tags_at_time(self, video_time: float) -> List[UWBTagPosition]:
        """
        Get UWB tag positions at specific video time.

        Args:
            video_time: Video time in seconds

        Returns:
            List of UWBTagPosition objects
        """
        # Get the first tag's first position to determine UWB start time
        if not self.tag_data:
            return []

        # Use the first available tag to get UWB log start time
        first_tag_id = list(self.tag_data.keys())[0]
        first_positions = self.tag_data[first_tag_id]
        if not first_positions:
            return []

        # Parse UWB log start datetime
        uwb_start = datetime.fromisoformat(first_positions[0]['datetime'])

        # Calculate target datetime: UWB start + video_time + sync_offset
        from datetime import timedelta
        target_dt = uwb_start + timedelta(seconds=(video_time + self.sync_offset))

        tags = []

        for tag_id, positions in self.tag_data.items():
            # Find closest position within tolerance
            closest_pos = None
            min_diff = float('inf')

            for pos in positions:
                # Parse datetime string to compare
                pos_time = datetime.fromisoformat(pos['datetime'])

                time_diff = abs((pos_time - target_dt).total_seconds())
                if time_diff < min_diff and time_diff < 5.0:  # 5 second tolerance (increased for debugging)
                    min_diff = time_diff
                    closest_pos = pos

            if closest_pos:
                x_uwb = closest_pos['x']
                y_uwb = closest_pos['y']

                # Filter out tags that are outside the court (using UWB coordinates)
                if not self.is_within_court_bounds(x_uwb, y_uwb):
                    continue

                # Convert to canvas pixels
                canvas_x, canvas_y = self.uwb_to_canvas_pixels(x_uwb, y_uwb)

                # Calculate calibration coordinates (for logging/consistency)
                # Use the same offset logic as project_player_to_court
                court_x = x_uwb + self.CALIB_OFFSET_X
                court_y = y_uwb + self.CALIB_OFFSET_Y

                tags.append(UWBTagPosition(
                    tag_id=tag_id,
                    x_uwb=x_uwb,
                    y_uwb=y_uwb,
                    court_x=court_x,
                    court_y=court_y,
                    canvas_x=canvas_x,
                    canvas_y=canvas_y,
                    datetime_str=closest_pos['datetime']
                ))

        return tags

    def detect_tagging_events(
        self,
        players: List[PlayerCourtPosition],
        tags: List[UWBTagPosition],
        frame_number: int,
        timestamp: float
    ) -> List[Tuple[PlayerCourtPosition, UWBTagPosition, float]]:
        """
        Detect tagging events (players within threshold of tags).

        Args:
            players: List of player positions
            tags: List of tag positions
            frame_number: Current frame number
            timestamp: Current video timestamp

        Returns:
            List of tuples (player, tag, distance) for tagging events
        """
        tagging_events = []

        for player in players:
            closest_tag = None
            min_distance = float('inf')

            for tag in tags:
                distance = self.calculate_distance_cm(
                    (player.court_x, player.court_y),
                    (tag.court_x, tag.court_y)
                )

                if distance <= self.config.tagging_threshold_cm and distance < min_distance:
                    min_distance = distance
                    closest_tag = tag

            if closest_tag:
                tagging_events.append((player, closest_tag, min_distance))

                # Log event
                event = TagEvent(
                    timestamp=timestamp,
                    frame_number=frame_number,
                    player_index=player.detection_index,
                    player_bbox=player.bbox,
                    player_court_x=player.court_x,
                    player_court_y=player.court_y,
                    tag_id=closest_tag.tag_id,
                    tag_x=closest_tag.x_uwb,
                    tag_y=closest_tag.y_uwb,
                    distance_cm=min_distance,
                    datetime_utc=closest_tag.datetime_str
                )
                self.event_logger.add_event(event)

        return tagging_events

    def draw_dotted_circle(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int]
    ):
        """
        Draw a dotted circle.

        Args:
            canvas: Image to draw on
            center: Circle center (x, y)
            radius: Circle radius in pixels
            color: BGR color
        """
        num_segments = 24  # 15° per segment

        for i in range(num_segments):
            if i % 2 == 0:  # Every other segment
                angle_start = i * 15
                angle_end = (i + 1) * 15

                try:
                    pts = cv2.ellipse2Poly(center, (radius, radius), 0,
                                          angle_start, angle_end, 5)
                    cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
                except Exception as e:
                    logger.debug(f"Failed to draw dotted circle segment: {e}")

    def draw_court_elements(
        self,
        canvas: np.ndarray,
        tags: List[UWBTagPosition],
        players: List[PlayerCourtPosition],
        tagging_events: List[Tuple[PlayerCourtPosition, UWBTagPosition, float]]
    ) -> np.ndarray:
        """
        Draw all court elements (tags, players, labels).

        Args:
            canvas: Court canvas image
            tags: List of UWB tag positions
            players: List of player positions
            tagging_events: List of tagging events

        Returns:
            Canvas with elements drawn
        """
        # Create tagged players set for quick lookup
        tagged_players = {event[0].detection_index: event[1] for event in tagging_events}

        # Calculate radius in pixels for 200cm circle
        radius_px = int(self.config.uwb_circle_radius_cm * self.scale_x)

        # Draw UWB tags
        for tag in tags:
            # Draw dotted radius circle
            self.draw_dotted_circle(
                canvas,
                (tag.canvas_x, tag.canvas_y),
                radius_px,
                self.config.uwb_tag_color
            )

            # Draw tag dot (larger)
            cv2.circle(
                canvas,
                (tag.canvas_x, tag.canvas_y),
                self.config.uwb_tag_radius_px + 5,
                self.config.uwb_tag_color,
                -1,  # Filled
                cv2.LINE_AA
            )

            # Draw white outline (thicker)
            cv2.circle(
                canvas,
                (tag.canvas_x, tag.canvas_y),
                self.config.uwb_tag_radius_px + 5,
                self.config.text_color,
                4,
                cv2.LINE_AA
            )

            # Draw tag ID
            tag_text = str(tag.tag_id)
            text_size, _ = cv2.getTextSize(
                tag_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 1.0,  # Larger font
                self.config.font_thickness + 1
            )

            text_x = tag.canvas_x - text_size[0] // 2
            text_y = tag.canvas_y - self.config.uwb_tag_radius_px - 20

            # Draw text background
            cv2.rectangle(
                canvas,
                (text_x - 6, text_y - text_size[1] - 6),
                (text_x + text_size[0] + 6, text_y + 6),
                self.config.text_bg_color,
                -1
            )

            # Draw text
            cv2.putText(
                canvas,
                tag_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 1.0,
                self.config.text_color,
                self.config.font_thickness + 1,
                cv2.LINE_AA
            )

        # Draw players
        for player in players:
            # Determine color based on tagging
            if player.detection_index in tagged_players:
                color = self.config.player_tagged_color
                tag = tagged_players[player.detection_index]
            else:
                color = self.config.player_untagged_color
                tag = None

            # Draw player dot (larger)
            cv2.circle(
                canvas,
                (player.canvas_x, player.canvas_y),
                self.config.player_radius_px + 10,
                color,
                -1,
                cv2.LINE_AA
            )

            # Draw white outline (thicker)
            cv2.circle(
                canvas,
                (player.canvas_x, player.canvas_y),
                self.config.player_radius_px + 10,
                self.config.text_color,
                4,
                cv2.LINE_AA
            )

            # Draw tag ID if tagged
            if tag:
                tag_text = f"Tag {tag.tag_id}"
                text_size, _ = cv2.getTextSize(
                    tag_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale * 1.0,
                    self.config.font_thickness + 1
                )

                text_x = player.canvas_x - text_size[0] // 2
                text_y = player.canvas_y + self.config.player_radius_px + text_size[1] + 20

                # Draw text background
                cv2.rectangle(
                    canvas,
                    (text_x - 6, text_y - text_size[1] - 6),
                    (text_x + text_size[0] + 6, text_y + 6),
                    self.config.text_bg_color,
                    -1
                )

                # Draw text
                cv2.putText(
                    canvas,
                    tag_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale * 1.0,
                    self.config.text_color,
                    self.config.font_thickness + 1,
                    cv2.LINE_AA
                )

        return canvas

    def process_frame(
        self,
        frame_number: int,
        video_frame: np.ndarray,
        video_time: float
    ) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame_number: Frame number
            video_frame: Video frame
            video_time: Video timestamp in seconds

        Returns:
            Court canvas with visualization
        """
        # Create fresh court canvas
        canvas = self.court_canvas_template.copy()

        # Detect players
        detections = self.player_detector.detect_players(video_frame)

        # Add index to detections
        for idx, detection in enumerate(detections):
            detection['index'] = idx

        # Project players to court
        players = []
        for detection in detections:
            player_pos = self.project_player_to_court(detection)
            if player_pos:
                players.append(player_pos)

        # Get tags at current time
        tags = self.get_tags_at_time(video_time)

        # Detect tagging events
        tagging_events = self.detect_tagging_events(
            players,
            tags,
            frame_number,
            video_time
        )

        # Draw all elements
        canvas = self.draw_court_elements(canvas, tags, players, tagging_events)

        # Draw info panel
        info_text = f"Frame: {frame_number} | Time: {video_time:.2f}s | Players: {len(players)} | Tags: {len(tags)} | Tagged: {len(tagging_events)}"
        cv2.putText(
            canvas,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.config.text_bg_color,
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.config.text_color,
            2,
            cv2.LINE_AA
        )

        return canvas

    def process_video(self) -> ProcessingResult:
        """
        Process entire video and generate output.

        Returns:
            ProcessingResult with status and paths
        """
        start_time = datetime.now()

        try:
            # Load court canvas template
            self.court_canvas_template = self.load_court_canvas()

            # Load tag data
            self.load_tag_data()

            # Calculate sync offset
            self.calculate_sync_offset()

            # Open video
            cap = cv2.VideoCapture(self.config.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.config.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine frame range
            start_frame = self.config.start_frame
            if self.config.end_frame:
                end_frame = min(self.config.end_frame, total_frames)
            elif self.config.max_frames:
                end_frame = min(start_frame + self.config.max_frames, total_frames)
            else:
                end_frame = total_frames

            logger.info(f"Processing frames {start_frame} to {end_frame} (FPS: {fps})")

            # Setup output paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path, events_path = self.config.get_output_paths(timestamp)

            # Create output directories
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            Path(events_path).parent.mkdir(parents=True, exist_ok=True)

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            out = cv2.VideoWriter(
                video_path,
                fourcc,
                self.config.output_fps,
                (self.canvas_width, self.canvas_height)
            )

            if not out.isOpened():
                raise RuntimeError(f"Failed to create video writer: {video_path}")

            # Set metadata for event logger
            metadata = ProcessingMetadata(
                video_file=self.config.video_path,
                sync_offset_seconds=self.sync_offset,
                tagging_threshold_cm=self.config.tagging_threshold_cm,
                processing_date=datetime.now().isoformat(),
                yolo_model=self.config.yolo_model,
                yolo_confidence=self.config.yolo_confidence,
                total_frames_processed=end_frame - start_frame,
                duration_seconds=(end_frame - start_frame) / fps
            )
            self.event_logger.set_metadata(metadata)

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Process frames
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_num}")
                    break

                video_time = frame_num / fps

                # Process frame
                canvas = self.process_frame(frame_num, frame, video_time)

                # Write to output
                out.write(canvas)

                # Log progress
                if (frame_num - start_frame) % 100 == 0:
                    progress = (frame_num - start_frame) / (end_frame - start_frame) * 100
                    logger.info(f"Progress: {progress:.1f}% (frame {frame_num}/{end_frame})")

            # Release resources
            cap.release()
            out.release()

            # Export events
            self.event_logger.export_to_json(events_path)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Video processing complete: {video_path}")
            logger.info(f"Events exported: {events_path}")
            logger.info(f"Total events: {len(self.event_logger.events)}")
            logger.info(f"Processing time: {processing_time:.1f} seconds")

            return ProcessingResult(
                success=True,
                output_video_path=video_path,
                output_events_path=events_path,
                events_count=len(self.event_logger.events),
                processing_time_seconds=processing_time
            )

        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
