"""
Association Logger for Persistent ID Tracking Pipeline.

Captures frame-by-frame association data during Pass 1:
- ByteTrack ID ↔ UWB Tag ID mappings
- Confidence levels and distances
- ID transition events (when ByteTrack loses/reassigns IDs)

Usage:
    logger = AssociationLogger(video_path, start_frame, end_frame, fps, sync_offset, proximity_threshold)

    # In frame loop
    logger.log_frame(frame_num, video_time, uwb_time, tracked_players, uwb_positions)

    # After processing
    logger.save('association_log.json')
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class AssociationLogger:
    """Logs frame-by-frame UWB association data for persistent ID tracking."""

    def __init__(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        fps: float,
        sync_offset: float,
        proximity_threshold: float
    ):
        """
        Initialize association logger.

        Args:
            video_path: Path to input video
            start_frame: Starting frame number
            end_frame: Ending frame number
            fps: Video frame rate
            sync_offset: UWB time synchronization offset (seconds)
            proximity_threshold: Association proximity threshold (pixels)
        """
        self.metadata = {
            "video": video_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": fps,
            "sync_offset": sync_offset,
            "proximity_threshold_px": proximity_threshold,
            "timestamp": datetime.now().isoformat()
        }

        self.frame_logs = []
        self.track_first_seen = {}  # track_id -> first frame
        self.track_last_seen = {}   # track_id -> last frame
        self.track_tag_mapping = {} # track_id -> tag_id (final known)
        self.transition_events = []

    def log_frame(
        self,
        frame_num: int,
        video_time: float,
        uwb_time: float,
        tracked_players: List[Dict],
        uwb_positions: Dict[int, Tuple[float, float]]
    ):
        """
        Log association data for a single frame.

        Args:
            frame_num: Current frame number
            video_time: Video timestamp (seconds)
            uwb_time: UWB timestamp (seconds)
            tracked_players: List of tracked player detections with UWB associations
            uwb_positions: Dict of {tag_id: (x_cm, y_cm)} UWB positions
        """
        associations = []

        for player in tracked_players:
            track_id = player['track_id']
            tag_id = player.get('uwb_tag_id')
            confidence = player.get('association_confidence', 'UNKNOWN')

            # Track first/last seen
            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = frame_num
            self.track_last_seen[track_id] = frame_num

            # Update final mapping and detect tag reassignments
            if tag_id is not None:
                old_tag = self.track_tag_mapping.get(track_id)
                if old_tag != tag_id and old_tag is not None:
                    # Tag reassignment event
                    self.transition_events.append({
                        "frame": frame_num,
                        "event": "TAG_REASSIGNED",
                        "track_id": track_id,
                        "old_tag_id": old_tag,
                        "new_tag_id": tag_id
                    })
                self.track_tag_mapping[track_id] = tag_id

            # Build association record
            assoc = {
                "track_id": track_id,
                "uwb_tag_id": tag_id,
                "confidence": confidence,
                "distance_px": player.get('association_distance'),
                "court_x": player.get('court_x'),
                "court_y": player.get('court_y')
            }

            # Add UWB position if tag is associated
            if tag_id and tag_id in uwb_positions:
                ux, uy = uwb_positions[tag_id]
                assoc["uwb_x"] = ux
                assoc["uwb_y"] = uy
            else:
                assoc["uwb_x"] = None
                assoc["uwb_y"] = None

            associations.append(assoc)

        # Log frame data
        self.frame_logs.append({
            "frame_num": frame_num,
            "video_time_sec": video_time,
            "uwb_time_sec": uwb_time,
            "associations": associations,
            "uwb_tags_visible": list(uwb_positions.keys()),
            "total_detections": len(tracked_players),
            "associated_count": sum(1 for p in tracked_players if p.get('uwb_tag_id'))
        })

    def detect_id_transitions(self):
        """
        Detect when ByteTrack loses/reassigns IDs.

        Analyzes tracking gaps > 50 frames (ByteTrack buffer limit).
        Generates transition events for lost and reacquired tracks.
        """
        for track_id, first_frame in self.track_first_seen.items():
            last_frame = self.track_last_seen[track_id]
            tag_id = self.track_tag_mapping.get(track_id)

            # Get all frames where this track was detected
            frames_seen = [
                log['frame_num'] for log in self.frame_logs
                if any(a['track_id'] == track_id for a in log['associations'])
            ]

            if len(frames_seen) < 2:
                continue

            # Check for gaps > 50 frames (ByteTrack buffer expired)
            for i in range(1, len(frames_seen)):
                gap = frames_seen[i] - frames_seen[i-1]
                if gap > 50:
                    self.transition_events.append({
                        "frame": frames_seen[i-1],
                        "event": "TRACK_LOST",
                        "track_id": track_id,
                        "tag_id": tag_id,
                        "gap_frames": gap,
                        "reason": "ByteTrack buffer expired (>50 frames)"
                    })

    def save(self, output_path: str):
        """
        Save association log to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        # Detect ID transitions
        self.detect_id_transitions()

        # Calculate summary statistics
        total_frames = len(self.frame_logs)
        total_assoc = sum(log['total_detections'] for log in self.frame_logs)
        successful_assoc = sum(log['associated_count'] for log in self.frame_logs)
        success_rate = (successful_assoc / total_assoc * 100) if total_assoc > 0 else 0

        # Build output structure
        output = {
            "metadata": self.metadata,
            "frames": self.frame_logs,
            "summary": {
                "total_frames": total_frames,
                "total_associations": total_assoc,
                "successful_associations": successful_assoc,
                "success_rate": round(success_rate, 2),
                "unique_track_ids": len(self.track_first_seen),
                "unique_uwb_tags": len(set(self.track_tag_mapping.values())),
                "track_to_tag_final_mapping": {
                    str(k): v for k, v in self.track_tag_mapping.items()
                },
                "id_transition_events": self.transition_events
            }
        }

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✅ Association log saved: {output_path}")
        print(f"   Total frames: {total_frames}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Unique tracks: {len(self.track_first_seen)}")
        print(f"   Unique UWB tags: {len(set(self.track_tag_mapping.values()))}")
        print(f"   Transition events: {len(self.transition_events)}")
