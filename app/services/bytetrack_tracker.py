"""ByteTrack Tracker for Basketball Player Tracking

Enhanced version supporting both YOLO bounding boxes and optional SAM2 masks.
Provides persistent tracking IDs across frames using motion-based matching
with optional mask-based features for improved accuracy.
"""

import numpy as np
from typing import List, Dict, Optional
import supervision as sv
import logging

logger = logging.getLogger(__name__)


class ByteTrackTracker:
    """ByteTrack tracker for maintaining persistent player IDs"""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 50,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        use_mask_features: bool = False
    ):
        """
        Initialize ByteTrack tracker

        Args:
            track_activation_threshold: Confidence threshold for track activation
            lost_track_buffer: Number of frames to keep lost tracks
            minimum_matching_threshold: IoU threshold for matching
            frame_rate: Video frame rate
            use_mask_features: Whether to use SAM2 mask features for enhanced tracking
        """
        # Initialize base ByteTrack tracker from supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )

        # Basketball-specific filtering
        self.min_box_area = 1000
        self.max_box_area = 50000
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 1.2

        # Mask feature support
        self.use_mask_features = use_mask_features

        # Tracking statistics
        self.frame_count = 0
        self.track_history = {}  # track_id -> {'positions': [], 'first_frame': int, ...}

        logger.info(f"ByteTrack tracker initialized")
        logger.info(f"  Activation threshold: {track_activation_threshold}")
        logger.info(f"  Lost track buffer: {lost_track_buffer} frames")
        logger.info(f"  Min matching threshold: {minimum_matching_threshold}")
        logger.info(f"  Mask-based features: {use_mask_features}")

    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections from YOLO with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - center: (cx, cy)
                - bottom: (bx, by)
            frame: Optional frame (not used but kept for API compatibility)

        Returns:
            List of tracked players with persistent IDs
        """
        self.frame_count += 1

        if not detections:
            return []

        # Filter detections for valid basketball players
        filtered_detections = self._filter_detections(detections)

        if not filtered_detections:
            return []

        # Convert to supervision format
        sv_detections = self._convert_to_supervision(filtered_detections)

        # Update base tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Convert back to our format
        tracked_players = self._convert_from_supervision(
            tracked_detections, filtered_detections
        )

        # Update tracking history
        self._update_history(tracked_players)

        return tracked_players

    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections for valid basketball players based on size and aspect ratio"""
        filtered = []

        for detection in detections:
            bbox = detection['bbox']

            # Calculate box properties
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0

            # Filter by size and aspect ratio
            if (self.min_box_area <= area <= self.max_box_area and
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                filtered.append(detection)

        return filtered

    def _convert_to_supervision(self, detections: List[Dict]) -> sv.Detections:
        """Convert our detection format to supervision format"""
        if not detections:
            return sv.Detections.empty()

        # Extract bounding boxes in xyxy format
        xyxy = np.array([det['bbox'] for det in detections], dtype=np.float32)

        # Extract confidences
        confidence = np.array([det['confidence'] for det in detections], dtype=np.float32)

        # All detections are class 0 (person)
        class_id = np.zeros(len(detections), dtype=int)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

    def _convert_from_supervision(
        self,
        sv_detections: sv.Detections,
        original_detections: List[Dict]
    ) -> List[Dict]:
        """Convert supervision format back to our format with tracking IDs and mask features"""
        tracked_players = []

        if len(sv_detections) == 0:
            return tracked_players

        for i in range(len(sv_detections)):
            # Get original detection data
            original_idx = i if i < len(original_detections) else 0
            original_det = original_detections[original_idx]

            # Get tracking ID from supervision
            track_id = int(sv_detections.tracker_id[i])
            bbox = sv_detections.xyxy[i].tolist()

            # Use mask centroid if available and mask features enabled
            if self.use_mask_features and 'mask_centroid' in original_det:
                center = list(original_det['mask_centroid'])
                # Use refined bbox bottom if available
                if 'refined_bbox' in original_det:
                    refined_bbox = original_det['refined_bbox']
                    bottom = [
                        (refined_bbox[0] + refined_bbox[2]) / 2,
                        refined_bbox[3]
                    ]
                else:
                    bottom = [center[0], bbox[3]]
            else:
                # Use traditional bbox center/bottom
                center = [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ]
                bottom = [
                    (bbox[0] + bbox[2]) / 2,
                    bbox[3]
                ]

            # Create tracked player with ID
            tracked_player = {
                'track_id': track_id,
                'bbox': bbox,
                'confidence': float(sv_detections.confidence[i]),
                'class_id': int(sv_detections.class_id[i]),
                'center': center,
                'bottom': bottom,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'frame_count': self.frame_count
            }

            # Copy over mask data if available
            for key in ['mask', 'refined_bbox', 'mask_centroid', 'mask_area', 'mask_confidence']:
                if key in original_det:
                    tracked_player[key] = original_det[key]

            # Copy over any additional data from original detection
            for key in ['uwb_tag_id', 'court_x', 'court_y']:
                if key in original_det:
                    tracked_player[key] = original_det[key]

            tracked_players.append(tracked_player)

        return tracked_players

    def _update_history(self, tracked_players: List[Dict]):
        """Update tracking history for statistics and analysis"""
        current_ids = set()

        for player in tracked_players:
            track_id = player['track_id']
            current_ids.add(track_id)

            # Initialize history for new tracks
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'first_frame': self.frame_count,
                    'last_frame': self.frame_count,
                    'total_frames': 1,
                    'positions': [player['center']],
                    'bboxes': [player['bbox']]
                }
            else:
                # Update existing track history
                history = self.track_history[track_id]
                history['last_frame'] = self.frame_count
                history['total_frames'] += 1
                history['positions'].append(player['center'])
                history['bboxes'].append(player['bbox'])

                # Keep only recent positions (last 30 frames)
                if len(history['positions']) > 30:
                    history['positions'] = history['positions'][-30:]
                    history['bboxes'] = history['bboxes'][-30:]

                # Add age to player info
                player['age'] = history['total_frames']

        # Clean up old tracks (every 100 frames)
        if self.frame_count % 100 == 0:
            old_ids = []
            for track_id, history in self.track_history.items():
                # Remove tracks not seen in 100 frames
                if self.frame_count - history['last_frame'] > 100:
                    old_ids.append(track_id)

            for track_id in old_ids:
                del self.track_history[track_id]

    def get_track_statistics(self) -> Dict:
        """Get tracking statistics"""
        active_tracks = len([
            track_id for track_id, history in self.track_history.items()
            if self.frame_count - history['last_frame'] < 10
        ])

        return {
            'frame_count': self.frame_count,
            'total_tracks': len(self.track_history),
            'active_tracks': active_tracks,
            'track_history': dict(self.track_history)
        }

    def reset(self):
        """Reset tracker state"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=50,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        self.frame_count = 0
        self.track_history = {}
        print("ByteTrack tracker reset")
