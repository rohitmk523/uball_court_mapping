"""ByteTrack tracker service for multi-object tracking."""
import numpy as np
from typing import Dict, List

from app.core.config import MATCH_THRESH, TRACK_BUFFER
from app.core.models import Detection, Track

# Try to use Ultralytics built-in tracker
try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("Warning: ByteTrack not available from Ultralytics")


class PlayerTracker:
    """Multi-object tracker for players using ByteTrack algorithm."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = TRACK_BUFFER,
        match_thresh: float = MATCH_THRESH
    ):
        """
        Initialize player tracker.

        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracker = None
        self.frame_id = 0
        self.tracks_dict = {}  # Manual tracking if ByteTrack not available

    def reset(self):
        """Reset tracker state."""
        self.tracker = None
        self.frame_id = 0
        self.tracks_dict = {}

    def update_tracks(self, detections: List[Detection], frame_id: int = None) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects
            frame_id: Current frame ID (optional)

        Returns:
            List of Track objects with track IDs
        """
        if frame_id is not None:
            self.frame_id = frame_id
        else:
            self.frame_id += 1

        # Use simple ID assignment if ByteTrack not available
        if not TRACKER_AVAILABLE:
            return self._simple_tracking(detections)

        # Convert detections to format expected by tracker
        # For now, use simple ID assignment based on detection order
        # In production, would integrate proper ByteTrack
        tracks = []
        for i, det in enumerate(detections):
            track = Track(
                track_id=i + 1,  # Simple sequential IDs
                bbox=det.bbox,
                confidence=det.confidence,
                bbox_bottom_center=det.bbox_bottom_center
            )
            tracks.append(track)

        return tracks

    def _simple_tracking(self, detections: List[Detection]) -> List[Track]:
        """
        Simple tracking fallback using position-based matching.

        Args:
            detections: List of detections

        Returns:
            List of tracks with IDs
        """
        tracks = []

        # Match detections to existing tracks based on position
        for det in detections:
            # Find closest existing track
            best_track_id = None
            best_distance = float('inf')

            for track_id, prev_track in self.tracks_dict.items():
                # Calculate distance between detection and previous track position
                det_x, det_y = det.bbox_bottom_center
                prev_x, prev_y = prev_track.bbox_bottom_center
                distance = np.sqrt((det_x - prev_x)**2 + (det_y - prev_y)**2)

                if distance < best_distance and distance < 100:  # 100 pixel threshold
                    best_distance = distance
                    best_track_id = track_id

            # Assign ID
            if best_track_id is not None:
                track_id = best_track_id
            else:
                # Create new track
                track_id = max(self.tracks_dict.keys(), default=0) + 1

            track = Track(
                track_id=track_id,
                bbox=det.bbox,
                confidence=det.confidence,
                bbox_bottom_center=det.bbox_bottom_center
            )

            tracks.append(track)
            self.tracks_dict[track_id] = track

        return tracks

    def track_video(
        self,
        detections_per_frame: Dict[int, List[Detection]]
    ) -> Dict[int, List[Track]]:
        """
        Apply tracking across all frames.

        Args:
            detections_per_frame: Dictionary of frame_id -> detections

        Returns:
            Dictionary of frame_id -> tracks
        """
        self.reset()
        tracks_per_frame = {}

        for frame_id in sorted(detections_per_frame.keys()):
            detections = detections_per_frame[frame_id]
            tracks = self.update_tracks(detections, frame_id)
            tracks_per_frame[frame_id] = tracks

        return tracks_per_frame


# Global tracker instance
_tracker = None


def get_tracker(
    track_thresh: float = 0.5,
    track_buffer: int = TRACK_BUFFER,
    match_thresh: float = MATCH_THRESH
) -> PlayerTracker:
    """
    Get global tracker instance (singleton pattern).

    Args:
        track_thresh: Detection confidence threshold
        track_buffer: Frame buffer for lost tracks
        match_thresh: IOU matching threshold

    Returns:
        PlayerTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = PlayerTracker(track_thresh, track_buffer, match_thresh)
    return _tracker
