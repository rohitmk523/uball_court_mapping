"""UWB Tag Associator for Player Tracking

Associates tracked player IDs from video with UWB tag IDs using spatial proximity.
Maintains fixed mapping (Track ID → Tag ID) for stable identification.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from bisect import bisect_left
import logging

logger = logging.getLogger(__name__)

# Import DXF parser for coordinate transformation
from app.services.dxf_parser import parse_court_dxf


class UWBAssociator:
    """Associates tracked players with UWB tags using spatial proximity"""

    def __init__(
        self,
        tags_dir: Path,
        sync_offset_seconds: float,
        uwb_start_time: Optional[datetime] = None,
        proximity_threshold: float = 200.0,  # Pixels (increased from 2.0 meters)
        validation_window: float = 5.0,
        remapping_cooldown: int = 30,
        court_dxf: Path = Path("court_2.dxf"),
        canvas_width: int = 4261,  # Vertical canvas width
        canvas_height: int = 7341,  # Vertical canvas height
        # Data cleaning parameters
        enable_uwb_cleaning: bool = True,
        max_velocity_cm_s: float = 1000.0,  # 10 m/s max speed
        gap_interpolation_max_s: float = 1.0,  # Max 1 second gaps
        smoothing_window_size: int = 5  # 5-frame moving average
    ):
        """
        Initialize UWB Associator

        Args:
            tags_dir: Directory containing UWB tag JSON files
            sync_offset_seconds: Sync offset between video and UWB (video_time + offset = uwb_time)
            uwb_start_time: UWB log start time (auto-detected if None)
            proximity_threshold: Maximum distance (pixels) for association
            validation_window: Time window (seconds) for validating existing mappings
            remapping_cooldown: Minimum frames before allowing remapping
            court_dxf: Path to court DXF file for coordinate transformation
            canvas_width: Canvas width (Vertical)
            canvas_height: Canvas height (Vertical)
            enable_uwb_cleaning: Enable UWB data cleaning (outlier rejection, interpolation, smoothing)
            max_velocity_cm_s: Maximum physically possible velocity in cm/s (default 1000 = 10 m/s)
            gap_interpolation_max_s: Maximum temporal gap to interpolate (seconds)
            smoothing_window_size: Moving average window size (frames)
        """
        self.tags_dir = Path(tags_dir)
        self.sync_offset = sync_offset_seconds
        self.proximity_threshold = proximity_threshold
        self.validation_window = validation_window
        self.remapping_cooldown = remapping_cooldown

        # Data cleaning parameters
        self.enable_uwb_cleaning = enable_uwb_cleaning
        self.max_velocity_cm_s = max_velocity_cm_s
        self.gap_interpolation_max_s = gap_interpolation_max_s
        self.smoothing_window_size = smoothing_window_size

        # Statistics tracking for data cleaning
        self.cleaning_stats = {
            'total_points': 0,
            'outliers_rejected': 0,
            'gaps_filled': 0,
            'total_interpolated_points': 0
        }

        # Load court geometry for UWB coordinate transformation
        self.court_dxf = court_dxf
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self._setup_coordinate_transform()

        # Persistent Track ID → Tag ID mapping
        self.track_to_tag_mapping = {}

        # Mapping metadata
        self.mapping_confidence = {}  # track_id -> confidence score
        self.mapping_age = {}  # track_id -> frames since mapping
        self.last_association_time = {}  # track_id -> last successful association frame

        # Load and preprocess UWB data
        self.tag_data = self._load_uwb_data()
        self.uwb_start_time = uwb_start_time or self._detect_uwb_start_time()
        self.sorted_tag_data = self._sort_and_index_uwb_data()

        # Statistics
        self.frame_count = 0
        self.total_associations = 0
        self.successful_associations = 0

        logger.info(f"UWB Associator initialized")
        logger.info(f"  Tags loaded: {len(self.tag_data)}")
        logger.info(f"  Sync offset: {sync_offset_seconds}s")
        logger.info(f"  Proximity threshold: {proximity_threshold}px")
        logger.info(f"  UWB start time: {self.uwb_start_time}")

    def _setup_coordinate_transform(self):
        """
        Setup UWB world-to-screen coordinate transformation.
        Matches logic in validation_overlap.py for Vertical Canvas.
        """
        # Parse court DXF to get world bounds
        geometry = parse_court_dxf(self.court_dxf)
        self.court_bounds = geometry.bounds

        # Canvas Dimensions (Vertical)
        # canvas_width (~4261) corresponds to Short Side of Court
        # canvas_height (~7341) corresponds to Long Side of Court
        
        court_width = self.court_bounds.max_x - self.court_bounds.min_x  # Long side (X)
        court_height = self.court_bounds.max_y - self.court_bounds.min_y # Short side (Y)
        
        PADDING = 50
        
        # Scale X(Long) to Canvas Height
        scale_y = (self.canvas_height - PADDING * 2) / court_width
        
        # Scale Y(Short) to Canvas Width
        scale_x = (self.canvas_width - PADDING * 2) / court_height
        
        self.scale = min(scale_x, scale_y)
        
        scaled_long = court_width * self.scale
        scaled_short = court_height * self.scale
        
        # Centering offsets
        self.offset_y = (self.canvas_height - scaled_long) / 2 # Vertical padding
        self.offset_x = (self.canvas_width - scaled_short) / 2 # Horizontal padding

        logger.debug(f"Coordinate transform setup: scale={self.scale:.6f}")

    def _world_to_screen(self, x_world: float, y_world: float) -> Tuple[float, float]:
        """
        Convert UWB world coordinates to screen coordinates (Vertical Canvas).
        """
        # Vertical Y (matches Long dimension X = x_world)
        sy = ((x_world - self.court_bounds.min_x) * self.scale) + self.offset_y
        
        # Vertical X (matches Short dimension Y = y_world)
        sx = ((y_world - self.court_bounds.min_y) * self.scale) + self.offset_x
        
        return int(sx), int(sy)

    def _load_uwb_data(self) -> Dict:
        """Load UWB tag data from JSON files"""
        tag_data = {}

        if not self.tags_dir.exists():
            logger.error(f"Tags directory not found: {self.tags_dir}")
            return tag_data

        for tag_file in self.tags_dir.glob('*.json'):
            try:
                with open(tag_file, 'r') as f:
                    data = json.load(f)
                    tag_id = data['tag_id']
                    tag_data[tag_id] = data['positions']
                    logger.debug(f"Loaded tag {tag_id}: {len(data['positions'])} positions")
            except Exception as e:
                logger.error(f"Failed to load {tag_file}: {e}")

        return tag_data

    def _detect_uwb_start_time(self) -> Optional[datetime]:
        """Auto-detect UWB log start time from first timestamp"""
        if not self.tag_data:
            return None

        # Get first tag's first position timestamp
        first_tag_id = list(self.tag_data.keys())[0]
        first_position = self.tag_data[first_tag_id][0]
        return datetime.fromisoformat(first_position['datetime'])

    def _sort_and_index_uwb_data(self) -> Dict:
        """Sort UWB positions by timestamp and apply data cleaning"""
        sorted_data = {}

        for tag_id, positions in self.tag_data.items():
            # Convert to (datetime, x, y) tuples and sort
            sorted_positions = []
            for pos in positions:
                dt = datetime.fromisoformat(pos['datetime'])
                sorted_positions.append((dt, pos['x'], pos['y']))

            sorted_positions.sort(key=lambda x: x[0])

            # Apply preprocessing if enabled
            if self.enable_uwb_cleaning and sorted_positions:
                sorted_positions = self._preprocess_uwb_trajectory(
                    tag_id, sorted_positions
                )

            sorted_data[tag_id] = sorted_positions

        # Log statistics
        if self.enable_uwb_cleaning:
            self._log_cleaning_statistics()

        return sorted_data

    def _reject_outliers(
        self,
        tag_id: int,
        positions: List[Tuple[datetime, float, float]]
    ) -> List[Tuple[datetime, float, float]]:
        """
        Reject positions with physically impossible velocities.

        Strategy:
        - Calculate velocity from previous valid point
        - Mark as outlier if velocity > max_velocity_cm_s (1000 cm/s = 10 m/s)
        - Keep first point (no previous reference)

        Edge Cases:
        - First point: Always kept (no reference)
        - Consecutive outliers: Each tested against last VALID point
        - All points rejected: Return list with only first point

        Args:
            tag_id: UWB tag identifier
            positions: Sorted list of (datetime, x_cm, y_cm) tuples

        Returns:
            Cleaned trajectory with outliers removed
        """
        if len(positions) <= 1:
            return positions

        cleaned = []
        outlier_count = 0

        # Always keep first point as reference
        cleaned.append(positions[0])
        last_valid_dt, last_valid_x, last_valid_y = positions[0]

        for i in range(1, len(positions)):
            curr_dt, curr_x, curr_y = positions[i]

            # Calculate time difference in seconds
            time_diff = (curr_dt - last_valid_dt).total_seconds()

            # Skip if time_diff is zero or negative (duplicate timestamps)
            if time_diff <= 0:
                outlier_count += 1
                continue

            # Calculate distance in cm
            distance_cm = np.sqrt(
                (curr_x - last_valid_x)**2 +
                (curr_y - last_valid_y)**2
            )

            # Calculate velocity in cm/s
            velocity_cm_s = distance_cm / time_diff

            # Check if velocity is physically possible
            if velocity_cm_s <= self.max_velocity_cm_s:
                # Valid point
                cleaned.append(positions[i])
                last_valid_dt, last_valid_x, last_valid_y = curr_dt, curr_x, curr_y
            else:
                # Outlier detected
                outlier_count += 1
                logger.debug(
                    f"Tag {tag_id}: Outlier at {curr_dt} "
                    f"(velocity={velocity_cm_s:.1f} cm/s, "
                    f"distance={distance_cm:.1f} cm, dt={time_diff:.3f}s)"
                )

        self.cleaning_stats['outliers_rejected'] += outlier_count

        if outlier_count > 0:
            logger.info(
                f"Tag {tag_id}: Rejected {outlier_count} outliers "
                f"({outlier_count/len(positions)*100:.1f}%)"
            )

        return cleaned

    def _interpolate_gaps(
        self,
        tag_id: int,
        positions: List[Tuple[datetime, float, float]]
    ) -> List[Tuple[datetime, float, float]]:
        """
        Fill temporal gaps with linearly interpolated positions.

        Strategy:
        - Detect gaps > 1 frame (~0.066s) but < gap_interpolation_max_s (1.0s)
        - Calculate expected frame times at ~15.74 Hz
        - Linearly interpolate (x, y) at missing timestamps

        Edge Cases:
        - Gaps > 1.0s: Not interpolated (likely tracking loss)
        - Gaps < 1 frame: No interpolation needed
        - Last position: No interpolation (no next point)

        Args:
            tag_id: UWB tag identifier
            positions: Sorted list of (datetime, x_cm, y_cm) tuples

        Returns:
            Trajectory with interpolated positions
        """
        if len(positions) <= 1:
            return positions

        EXPECTED_FRAME_INTERVAL = 0.066  # ~15 Hz (1/15.74)
        FRAME_TOLERANCE = 0.01  # 10ms tolerance

        interpolated = []
        total_gaps_filled = 0
        total_points_added = 0

        for i in range(len(positions) - 1):
            curr_dt, curr_x, curr_y = positions[i]
            next_dt, next_x, next_y = positions[i + 1]

            # Always keep current point
            interpolated.append(positions[i])

            # Calculate gap duration
            gap_duration = (next_dt - curr_dt).total_seconds()

            # Check if gap needs interpolation
            if gap_duration > (EXPECTED_FRAME_INTERVAL + FRAME_TOLERANCE):
                if gap_duration <= self.gap_interpolation_max_s:
                    # Calculate number of missing frames
                    missing_frames = int(gap_duration / EXPECTED_FRAME_INTERVAL) - 1

                    if missing_frames > 0:
                        # Linear interpolation
                        for frame_idx in range(1, missing_frames + 1):
                            # Interpolation factor [0, 1]
                            alpha = frame_idx / (missing_frames + 1)

                            # Interpolated timestamp
                            interp_dt = curr_dt + timedelta(
                                seconds=gap_duration * alpha
                            )

                            # Interpolated position
                            interp_x = curr_x + (next_x - curr_x) * alpha
                            interp_y = curr_y + (next_y - curr_y) * alpha

                            interpolated.append((interp_dt, interp_x, interp_y))
                            total_points_added += 1

                        total_gaps_filled += 1
                else:
                    # Gap too large - likely tracking dropout
                    logger.debug(
                        f"Tag {tag_id}: Large gap at {curr_dt} "
                        f"(duration={gap_duration:.2f}s) - not interpolating"
                    )

        # Add last point
        interpolated.append(positions[-1])

        self.cleaning_stats['gaps_filled'] += total_gaps_filled
        self.cleaning_stats['total_interpolated_points'] += total_points_added

        if total_gaps_filled > 0:
            logger.info(
                f"Tag {tag_id}: Filled {total_gaps_filled} gaps "
                f"(added {total_points_added} interpolated points)"
            )

        return interpolated

    def _apply_smoothing(
        self,
        tag_id: int,
        positions: List[Tuple[datetime, float, float]]
    ) -> List[Tuple[datetime, float, float]]:
        """
        Apply moving average filter to reduce jitter.

        Strategy:
        - Use centered window of size smoothing_window_size (default 5)
        - Window: [i-2, i-1, i, i+1, i+2] for center point i
        - Edge handling: Use asymmetric windows at boundaries

        Edge Cases:
        - Start boundary (i < window_size//2): Use forward window [i, i+window]
        - End boundary (i > len-window_size//2): Use backward window [i-window, i]
        - Too few points (< window_size): Return unchanged

        Args:
            tag_id: UWB tag identifier
            positions: Sorted list of (datetime, x_cm, y_cm) tuples

        Returns:
            Smoothed trajectory
        """
        if len(positions) < self.smoothing_window_size:
            return positions

        window_radius = self.smoothing_window_size // 2
        smoothed = []

        for i in range(len(positions)):
            curr_dt, curr_x, curr_y = positions[i]

            # Determine window bounds
            if i < window_radius:
                # Start boundary: forward window
                start_idx = 0
                end_idx = min(self.smoothing_window_size, len(positions))
            elif i >= len(positions) - window_radius:
                # End boundary: backward window
                start_idx = max(0, len(positions) - self.smoothing_window_size)
                end_idx = len(positions)
            else:
                # Center: symmetric window
                start_idx = i - window_radius
                end_idx = i + window_radius + 1

            # Calculate average position in window
            window_positions = positions[start_idx:end_idx]
            avg_x = sum(x for dt, x, y in window_positions) / len(window_positions)
            avg_y = sum(y for dt, x, y in window_positions) / len(window_positions)

            # Keep original timestamp, use averaged position
            smoothed.append((curr_dt, avg_x, avg_y))

        logger.debug(f"Tag {tag_id}: Applied smoothing (window={self.smoothing_window_size})")

        return smoothed

    def _preprocess_uwb_trajectory(
        self,
        tag_id: int,
        positions: List[Tuple[datetime, float, float]]
    ) -> List[Tuple[datetime, float, float]]:
        """
        Apply three-stage cleaning to UWB trajectory:
        1. Outlier rejection (velocity-based)
        2. Gap interpolation (temporal)
        3. Smoothing (moving average)

        Args:
            tag_id: UWB tag identifier
            positions: Sorted list of (datetime, x_cm, y_cm) tuples

        Returns:
            Cleaned trajectory as list of (datetime, x_cm, y_cm) tuples
        """
        if not positions:
            return positions

        # Track original count
        self.cleaning_stats['total_points'] += len(positions)

        # Stage 1: Reject outliers
        positions = self._reject_outliers(tag_id, positions)

        # Stage 2: Interpolate gaps
        positions = self._interpolate_gaps(tag_id, positions)

        # Stage 3: Apply smoothing
        positions = self._apply_smoothing(tag_id, positions)

        return positions

    def _log_cleaning_statistics(self):
        """Log summary of data cleaning operations"""
        stats = self.cleaning_stats

        if stats['total_points'] == 0:
            return

        outlier_pct = (stats['outliers_rejected'] / stats['total_points']) * 100
        final_count = stats['total_points'] - stats['outliers_rejected'] + stats['total_interpolated_points']

        logger.info("=" * 60)
        logger.info("UWB Data Cleaning Statistics:")
        logger.info(f"  Total input points:     {stats['total_points']:,}")
        logger.info(f"  Outliers rejected:      {stats['outliers_rejected']:,} ({outlier_pct:.2f}%)")
        logger.info(f"  Gaps filled:            {stats['gaps_filled']:,}")
        logger.info(f"  Interpolated points:    {stats['total_interpolated_points']:,}")
        logger.info(f"  Final point count:      {final_count:,}")
        logger.info("=" * 60)

    def associate(
        self,
        frame_time: float,
        tracked_players: List[Dict]
    ) -> List[Dict]:
        """
        Associate tracked players with UWB tags

        Args:
            frame_time: Video timestamp in seconds
            tracked_players: List of tracked players with court coordinates
                Expected keys: 'track_id', 'court_x', 'court_y', ...

        Returns:
            Tracked players with added 'uwb_tag_id' and 'association_confidence'
        """
        self.frame_count += 1

        if not tracked_players:
            return []

        # Get UWB positions at synchronized time
        uwb_time = frame_time + self.sync_offset
        target_dt = self.uwb_start_time + timedelta(seconds=uwb_time)
        uwb_positions = self._get_uwb_positions_at_time(target_dt)

        # Associate each tracked player with a UWB tag
        for player in tracked_players:
            track_id = player.get('track_id')
            if track_id is None:
                logger.warning("Player missing track_id, skipping association")
                continue

            # Get player position in court coordinates
            player_x = player.get('court_x')
            player_y = player.get('court_y')

            if player_x is None or player_y is None:
                # logger.warning(f"Track {track_id} missing court coordinates") 
                # Reduced logging to avoid spam
                player['uwb_tag_id'] = None
                player['association_confidence'] = 'NO_COORDS'
                continue

            # Check if we have an existing mapping
            if track_id in self.track_to_tag_mapping:
                tag_id = self.track_to_tag_mapping[track_id]

                # Validate existing mapping
                if self._validate_mapping(player_x, player_y, tag_id, uwb_positions):
                    player['uwb_tag_id'] = tag_id
                    player['association_confidence'] = 'HIGH'
                    self.mapping_age[track_id] = self.mapping_age.get(track_id, 0) + 1
                    self.last_association_time[track_id] = self.frame_count
                    self.successful_associations += 1
                    continue

                # Existing mapping failed validation
                logger.debug(f"Track {track_id} mapping to tag {tag_id} failed validation")

                # Only remap if cooldown elapsed
                if self.mapping_age.get(track_id, 0) < self.remapping_cooldown:
                    player['uwb_tag_id'] = tag_id
                    player['association_confidence'] = 'WEAK'
                    continue

            # Find new association using spatial proximity
            best_tag_id, distance = self._find_nearest_tag(
                player_x, player_y, uwb_positions
            )

            if best_tag_id is not None and distance < self.proximity_threshold:
                # Create new mapping
                self.track_to_tag_mapping[track_id] = best_tag_id
                self.mapping_age[track_id] = 0
                self.last_association_time[track_id] = self.frame_count

                player['uwb_tag_id'] = best_tag_id
                player['association_confidence'] = 'NEW'
                player['association_distance'] = distance

                logger.debug(f"New mapping: Track {track_id} → Tag {best_tag_id} (distance: {distance:.2f}m)")
                self.total_associations += 1
                self.successful_associations += 1
            else:
                # No nearby UWB tag found
                player['uwb_tag_id'] = None
                player['association_confidence'] = 'LOW'
                player['association_distance'] = distance if best_tag_id else None

        return tracked_players

    def _get_uwb_positions_at_time(self, target_dt: datetime) -> Dict[int, Tuple[float, float]]:
        """
        Get UWB tag positions at a specific time using binary search.
        Returns positions in SCREEN coordinates (rotated canvas space) for comparison with homography output.
        """
        uwb_positions = {}

        for tag_id, sorted_positions in self.sorted_tag_data.items():
            if not sorted_positions:
                continue

            # Binary search for closest timestamp
            timestamps = [dt for dt, x, y in sorted_positions]
            idx = bisect_left(timestamps, target_dt)

            # Check adjacent positions to find closest
            candidates = []
            if idx > 0:
                time_diff = abs((timestamps[idx - 1] - target_dt).total_seconds())
                candidates.append((idx - 1, time_diff))
            if idx < len(timestamps):
                time_diff = abs((timestamps[idx] - target_dt).total_seconds())
                candidates.append((idx, time_diff))

            if candidates:
                best_idx, time_diff = min(candidates, key=lambda x: x[1])

                # Only use if within time threshold (5 seconds)
                if time_diff < 5.0:
                    dt, x_world, y_world = sorted_positions[best_idx]
                    # Transform world coordinates to screen coordinates
                    screen_x, screen_y = self._world_to_screen(x_world, y_world)
                    uwb_positions[tag_id] = (screen_x, screen_y)

        return uwb_positions

    def _find_nearest_tag(
        self,
        player_x: float,
        player_y: float,
        uwb_positions: Dict[int, Tuple[float, float]]
    ) -> Tuple[Optional[int], float]:
        """Find nearest UWB tag to player position"""
        if not uwb_positions:
            return None, float('inf')

        min_distance = float('inf')
        nearest_tag = None

        for tag_id, (tag_x, tag_y) in uwb_positions.items():
            # Calculate Euclidean distance
            distance = np.sqrt((player_x - tag_x)**2 + (player_y - tag_y)**2)

            if distance < min_distance:
                min_distance = distance
                nearest_tag = tag_id

        return nearest_tag, min_distance

    def _validate_mapping(
        self,
        player_x: float,
        player_y: float,
        tag_id: int,
        uwb_positions: Dict[int, Tuple[float, float]]
    ) -> bool:
        """Validate existing Track → Tag mapping"""
        if tag_id not in uwb_positions:
            return False

        tag_x, tag_y = uwb_positions[tag_id]
        distance = np.sqrt((player_x - tag_x)**2 + (player_y - tag_y)**2)

        # Use slightly larger threshold for validation
        validation_threshold = self.proximity_threshold * 1.5
        return distance < validation_threshold

    def get_statistics(self) -> Dict:
        """Get association statistics"""
        active_mappings = len(self.track_to_tag_mapping)
        success_rate = (self.successful_associations / self.total_associations * 100
                       if self.total_associations > 0 else 0)

        return {
            'frame_count': self.frame_count,
            'active_mappings': active_mappings,
            'total_associations_attempted': self.total_associations,
            'successful_associations': self.successful_associations,
            'success_rate_percent': success_rate,
            'track_to_tag_mapping': dict(self.track_to_tag_mapping),
            'mapping_ages': dict(self.mapping_age)
        }

    def reset(self):
        """Reset all mappings and statistics"""
        self.track_to_tag_mapping = {}
        self.mapping_confidence = {}
        self.mapping_age = {}
        self.last_association_time = {}
        self.frame_count = 0
        self.total_associations = 0
        self.successful_associations = 0
        logger.info("UWB Associator reset")
