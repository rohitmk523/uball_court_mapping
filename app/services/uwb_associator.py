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


class UWBAssociator:
    """Associates tracked players with UWB tags using spatial proximity"""

    def __init__(
        self,
        tags_dir: Path,
        sync_offset_seconds: float,
        uwb_start_time: Optional[datetime] = None,
        proximity_threshold: float = 2.0,
        validation_window: float = 5.0,
        remapping_cooldown: int = 30
    ):
        """
        Initialize UWB Associator

        Args:
            tags_dir: Directory containing UWB tag JSON files
            sync_offset_seconds: Sync offset between video and UWB (video_time + offset = uwb_time)
            uwb_start_time: UWB log start time (auto-detected if None)
            proximity_threshold: Maximum distance (meters) for association
            validation_window: Time window (seconds) for validating existing mappings
            remapping_cooldown: Minimum frames before allowing remapping
        """
        self.tags_dir = Path(tags_dir)
        self.sync_offset = sync_offset_seconds
        self.proximity_threshold = proximity_threshold
        self.validation_window = validation_window
        self.remapping_cooldown = remapping_cooldown

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
        logger.info(f"  Proximity threshold: {proximity_threshold}m")
        logger.info(f"  UWB start time: {self.uwb_start_time}")

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
        """Sort UWB positions by timestamp for efficient lookup"""
        sorted_data = {}

        for tag_id, positions in self.tag_data.items():
            # Convert to (datetime, x, y) tuples and sort
            sorted_positions = []
            for pos in positions:
                dt = datetime.fromisoformat(pos['datetime'])
                sorted_positions.append((dt, pos['x'], pos['y']))

            sorted_positions.sort(key=lambda x: x[0])
            sorted_data[tag_id] = sorted_positions

        return sorted_data

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
                logger.warning(f"Track {track_id} missing court coordinates")
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
                logger.warning(f"Track {track_id} mapping to tag {tag_id} failed validation")

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

                logger.debug(f"Track {track_id} no nearby tag (closest: {distance:.2f}m if distance else 'N/A')")

        return tracked_players

    def _get_uwb_positions_at_time(self, target_dt: datetime) -> Dict[int, Tuple[float, float]]:
        """Get UWB tag positions at a specific time using binary search"""
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
                    dt, x, y = sorted_positions[best_idx]
                    uwb_positions[tag_id] = (x, y)

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
