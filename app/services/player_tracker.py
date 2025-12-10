"""
Player tracker using IoU matching + UWB tag association.

Simplified ByteTrack-style approach:
- Track players frame-to-frame using IoU (Intersection over Union)
- Periodically re-match with UWB tags (every 2 seconds)
- Maintain persistent IDs across frames
- Tag players within 200cm of UWB tags (green), others stay red
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class TrackedPlayer:
    """Represents a tracked player with persistent ID"""

    def __init__(self, track_id: int, position: Tuple[float, float], bbox: List[float]):
        self.track_id = track_id
        self.position = position  # (x, y) in world coordinates (cm)
        self.bbox = bbox  # [x1, y1, x2, y2] in canvas pixels
        self.uwb_tag_id = None  # Associated UWB tag ID (if tagged)
        self.frames_lost = 0  # Frames since last detection
        self.is_active = True


class PlayerTracker:
    """
    Hybrid tracker: IoU matching + UWB association

    Maintains persistent player IDs using:
    1. Frame-to-frame IoU matching (for temporal consistency)
    2. Periodic UWB tag matching (every 2 seconds for ground truth alignment)
    """

    def __init__(
        self,
        uwb_sync_offset: float = 785.0,  # Sync offset in seconds
        tagging_radius_cm: float = 200.0,  # 200cm radius for UWB tagging
        rematch_interval: float = 2.0,  # Re-match with UWB every 2 seconds
        iou_threshold: float = 0.3,  # Min IoU for same player
        max_frames_lost: int = 30,  # Drop track after 30 frames (~1 second)
    ):
        self.uwb_sync_offset = uwb_sync_offset
        self.tagging_radius_cm = tagging_radius_cm
        self.rematch_interval = rematch_interval
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost

        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedPlayer] = {}
        self.last_rematch_time = -999.0  # Force initial match

    def update(
        self,
        detections: List[Dict],  # YOLO detections with 'bbox', 'position' keys
        uwb_positions: Dict[str, Tuple[float, float]],  # {tag_id: (x_cm, y_cm)}
        current_time: float,  # Video time in seconds
    ) -> List[TrackedPlayer]:
        """
        Update tracker with new detections and UWB positions.

        Returns list of tracked players with IDs and UWB associations.
        """
        # 1. Match detections with existing tracks using IoU
        matches, unmatched_detections, unmatched_tracks = self._match_detections(detections)

        # 2. Update matched tracks
        for det_idx, track_id in matches:
            detection = detections[det_idx]
            track = self.active_tracks[track_id]
            track.position = detection['position']
            track.bbox = detection['bbox']
            track.frames_lost = 0

        # 3. Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            new_track = TrackedPlayer(
                track_id=self.next_track_id,
                position=detection['position'],
                bbox=detection['bbox']
            )
            self.active_tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # 4. Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            self.active_tracks[track_id].frames_lost += 1

        # 5. Remove lost tracks
        self._remove_lost_tracks()

        # 6. Periodic UWB re-matching
        if current_time - self.last_rematch_time >= self.rematch_interval:
            self._rematch_with_uwb(uwb_positions)
            self.last_rematch_time = current_time

        # 7. Return active tracks
        return list(self.active_tracks.values())

    def _match_detections(
        self, detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections with existing tracks using IoU.

        Returns:
            - matches: List of (detection_idx, track_id) pairs
            - unmatched_detections: Detection indices with no match
            - unmatched_tracks: Track IDs with no match
        """
        if not self.active_tracks or not detections:
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(self.active_tracks.keys())
            return [], unmatched_dets, unmatched_tracks

        # Build cost matrix: IoU between each detection and track
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for det_idx, detection in enumerate(detections):
            det_bbox = detection['bbox']
            for track_idx, track_id in enumerate(track_ids):
                track_bbox = self.active_tracks[track_id].bbox
                iou = self._calculate_iou(det_bbox, track_bbox)
                cost_matrix[det_idx, track_idx] = 1.0 - iou  # Convert to cost

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by IoU threshold
        matches = []
        matched_det_indices = set()
        matched_track_indices = set()

        for det_idx, track_idx in zip(row_indices, col_indices):
            iou = 1.0 - cost_matrix[det_idx, track_idx]
            if iou >= self.iou_threshold:
                track_id = track_ids[track_idx]
                matches.append((det_idx, track_id))
                matched_det_indices.add(det_idx)
                matched_track_indices.add(track_idx)

        # Unmatched detections and tracks
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracks = [
            track_ids[i] for i in range(len(track_ids))
            if i not in matched_track_indices
        ]

        return matches, unmatched_dets, unmatched_tracks

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _remove_lost_tracks(self):
        """Remove tracks that haven't been detected for too long"""
        to_remove = [
            track_id for track_id, track in self.active_tracks.items()
            if track.frames_lost > self.max_frames_lost
        ]
        for track_id in to_remove:
            del self.active_tracks[track_id]

    def _rematch_with_uwb(self, uwb_positions: Dict[str, Tuple[float, float]]):
        """
        Re-match all active tracks with UWB tags using distance-based assignment.

        Updates track.uwb_tag_id for players within tagging radius.
        """
        if not self.active_tracks or not uwb_positions:
            # Clear all UWB associations if no UWB data
            for track in self.active_tracks.values():
                track.uwb_tag_id = None
            return

        # Build cost matrix: distance between each track and UWB tag
        track_ids = list(self.active_tracks.keys())
        uwb_tag_ids = list(uwb_positions.keys())

        cost_matrix = np.zeros((len(track_ids), len(uwb_tag_ids)))

        for track_idx, track_id in enumerate(track_ids):
            track = self.active_tracks[track_id]
            tx, ty = track.position

            for uwb_idx, uwb_tag_id in enumerate(uwb_tag_ids):
                ux, uy = uwb_positions[uwb_tag_id]
                distance = np.sqrt((tx - ux)**2 + (ty - uy)**2)
                cost_matrix[track_idx, uwb_idx] = distance

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Clear all UWB associations first
        for track in self.active_tracks.values():
            track.uwb_tag_id = None

        # Assign UWB tags to tracks within radius
        for track_idx, uwb_idx in zip(row_indices, col_indices):
            distance = cost_matrix[track_idx, uwb_idx]
            if distance <= self.tagging_radius_cm:
                track_id = track_ids[track_idx]
                uwb_tag_id = uwb_tag_ids[uwb_idx]
                self.active_tracks[track_id].uwb_tag_id = uwb_tag_id

    def get_tracking_stats(self) -> Dict:
        """Get current tracking statistics"""
        total_tracks = len(self.active_tracks)
        tagged_tracks = sum(1 for t in self.active_tracks.values() if t.uwb_tag_id is not None)

        return {
            'total_tracks': total_tracks,
            'tagged_tracks': tagged_tracks,
            'untagged_tracks': total_tracks - tagged_tracks,
            'next_track_id': self.next_track_id
        }
