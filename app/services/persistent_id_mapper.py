"""
Persistent ID Mapper for Two-Pass Tracking Pipeline.

Resolves ByteTrack ID instability by maintaining a mapping to UWB tag IDs.
Provides consistent color palette tied to UWB tags (not ByteTrack IDs).

Usage:
    # Pass 2: Load association log from Pass 1
    mapper = PersistentIDMapper('association_log.json')

    # For each frame
    tag_id = mapper.get_persistent_id(frame_num, track_id)
    color = mapper.get_color_for_tag(tag_id)
    label = mapper.get_label(track_id, tag_id, mode='dual')
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PersistentIDMapper:
    """Maps ByteTrack IDs to persistent UWB tag IDs with consistent colors."""

    def __init__(self, association_log_path: str):
        """
        Load association log and build persistent ID mapping.

        Args:
            association_log_path: Path to association log JSON (from Pass 1)
        """
        # Load association log
        with open(association_log_path, 'r') as f:
            data = json.load(f)

        self.metadata = data['metadata']
        self.frame_logs = {log['frame_num']: log for log in data['frames']}
        self.summary = data['summary']

        # Build frame-indexed mappings: {frame_num: {track_id: tag_id}}
        self.frame_mappings = {}
        for frame_num, log in self.frame_logs.items():
            self.frame_mappings[frame_num] = {}
            for assoc in log['associations']:
                track_id = assoc['track_id']
                tag_id = assoc['uwb_tag_id']
                if tag_id is not None:
                    self.frame_mappings[frame_num][track_id] = tag_id

        # Generate consistent color palette for UWB tags
        unique_tags = list(set(
            int(tag_id) for tag_id in self.summary['track_to_tag_final_mapping'].values()
        ))
        self.tag_to_color = self._generate_tag_colors(unique_tags)

        # Build tag history: which ByteTrack IDs were used for each tag
        self.tag_history = {}  # {tag_id: [track_ids]}
        for track_id_str, tag_id in self.summary['track_to_tag_final_mapping'].items():
            if tag_id not in self.tag_history:
                self.tag_history[tag_id] = []
            self.tag_history[tag_id].append(int(track_id_str))

        print(f"✅ PersistentIDMapper loaded: {association_log_path}")
        print(f"   Unique UWB tags: {len(unique_tags)}")
        print(f"   Color palette generated for {len(self.tag_to_color)} tags")

    def _generate_tag_colors(self, tag_ids: List[int]) -> Dict[int, Tuple[int, int, int]]:
        """
        Generate distinct colors for each UWB tag (consistent across frames).

        Args:
            tag_ids: List of unique UWB tag IDs

        Returns:
            Dict mapping tag_id → (B, G, R) color
        """
        colors = {}
        n = len(tag_ids)

        for i, tag_id in enumerate(sorted(tag_ids)):  # Sort for consistency
            # Distribute hues evenly across HSV spectrum
            hue = int(180 * i / max(n, 1))
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors[tag_id] = tuple(map(int, color_bgr))

        return colors

    def get_persistent_id(self, frame_num: int, track_id: int) -> Optional[int]:
        """
        Get UWB tag_id for a ByteTrack ID at a specific frame.

        Strategy:
        1. Direct lookup in frame mapping
        2. If not found, check recent frames (±30 frames / ~1 second)
        3. Return None if no association found

        Args:
            frame_num: Current frame number
            track_id: ByteTrack track ID

        Returns:
            UWB tag_id or None if unassociated
        """
        # Direct lookup
        if frame_num in self.frame_mappings:
            if track_id in self.frame_mappings[frame_num]:
                return self.frame_mappings[frame_num][track_id]

        # Fallback: check recent frames (±30 frames / ~1 second at 30fps)
        for offset in range(1, 31):
            for sign in [-1, 1]:
                check_frame = frame_num + (sign * offset)
                if check_frame in self.frame_mappings:
                    if track_id in self.frame_mappings[check_frame]:
                        return self.frame_mappings[check_frame][track_id]

        return None

    def get_color_for_tag(self, tag_id: Optional[int]) -> Tuple[int, int, int]:
        """
        Get consistent BGR color for UWB tag.

        Args:
            tag_id: UWB tag ID (or None for unassociated)

        Returns:
            BGR color tuple (gray for None)
        """
        if tag_id is None:
            return (128, 128, 128)  # Gray for unassociated
        return self.tag_to_color.get(tag_id, (255, 255, 255))  # White fallback

    def get_label(
        self,
        track_id: int,
        tag_id: Optional[int],
        mode: str = 'dual'
    ) -> str:
        """
        Generate display label based on mode.

        Args:
            track_id: ByteTrack ID
            tag_id: UWB tag ID (or None)
            mode: Display mode
                - 'dual': Show both IDs (e.g., "BT:5 Tag:1587672")
                - 'tag_only': Show only UWB tag (e.g., "Tag:1587672")
                - 'track_only': Show only ByteTrack ID (e.g., "ID:5")

        Returns:
            Formatted label string
        """
        if mode == 'dual':
            if tag_id is not None:
                return f"BT:{track_id} Tag:{tag_id}"
            else:
                return f"BT:{track_id}"
        elif mode == 'tag_only':
            return f"Tag:{tag_id}" if tag_id else f"ID:{track_id}"
        else:  # track_only
            return f"ID:{track_id}"

    def get_statistics(self) -> Dict:
        """
        Return summary statistics from association log.

        Returns:
            Summary dictionary with stats
        """
        return self.summary

    def generate_statistics_report(self, output_path: str, video_info: Dict = None):
        """
        Generate a detailed markdown statistics report.

        Args:
            output_path: Path to save the markdown report
            video_info: Optional dict with video metadata (fps, duration, etc.)
        """
        from datetime import datetime

        report_lines = []
        report_lines.append("# Persistent ID Tracking - Statistics Report")
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Video Information
        if video_info:
            report_lines.append("## Video Information")
            report_lines.append(f"- **Video**: {video_info.get('video_path', 'N/A')}")
            report_lines.append(f"- **Time Range**: {video_info.get('start_time', 'N/A')} to {video_info.get('end_time', 'N/A')}")
            report_lines.append(f"- **Duration**: {video_info.get('duration', 'N/A'):.1f}s")
            report_lines.append(f"- **Total Frames**: {video_info.get('total_frames', 'N/A')}")
            report_lines.append(f"- **FPS**: {video_info.get('fps', 'N/A'):.2f}\n")

        # Overall Statistics
        report_lines.append("## Overall Statistics")
        report_lines.append(f"- **Unique UWB Tags**: {self.summary.get('unique_uwb_tags', 0)}")
        report_lines.append(f"- **Total ByteTrack IDs**: {self.summary.get('total_track_ids', 0)}")
        report_lines.append(f"- **Association Success Rate**: {self.summary.get('success_rate', 0):.1f}%")
        report_lines.append(f"- **Frames with Associations**: {self.summary.get('frames_with_associations', 0)}")
        report_lines.append(f"- **Total Associations**: {self.summary.get('total_associations', 0)}\n")

        # Per-Tag Statistics
        report_lines.append("## Per-Tag Statistics")
        report_lines.append("\n| Tag ID | First Seen | Last Seen | Frames Visible | Associated Track IDs |")
        report_lines.append("|--------|------------|-----------|----------------|----------------------|")

        tag_stats = self.summary.get('per_tag_stats', {})
        for tag_id in sorted(tag_stats.keys()):
            stats = tag_stats[tag_id]
            first_frame = stats.get('first_frame', 'N/A')
            last_frame = stats.get('last_frame', 'N/A')
            frame_count = stats.get('frame_count', 0)
            track_ids = stats.get('track_ids', [])
            track_ids_str = ', '.join(map(str, sorted(track_ids)[:5]))
            if len(track_ids) > 5:
                track_ids_str += f" (+{len(track_ids)-5} more)"

            report_lines.append(
                f"| {tag_id} | {first_frame} | {last_frame} | {frame_count} | {track_ids_str} |"
            )

        # ID Transitions
        report_lines.append("\n## ID Transition Events")
        transitions = self.summary.get('id_transitions', [])
        if transitions:
            report_lines.append(f"\nTotal transition events: {len(transitions)}\n")
            report_lines.append("| Event # | Tag ID | Old Track ID | New Track ID | Gap (frames) | Frame |")
            report_lines.append("|---------|--------|--------------|--------------|--------------|-------|")
            for i, trans in enumerate(transitions[:20], 1):  # Show first 20
                report_lines.append(
                    f"| {i} | {trans.get('tag_id', 'N/A')} | "
                    f"{trans.get('old_track_id', 'N/A')} | "
                    f"{trans.get('new_track_id', 'N/A')} | "
                    f"{trans.get('gap_frames', 'N/A')} | "
                    f"{trans.get('frame', 'N/A')} |"
                )
            if len(transitions) > 20:
                report_lines.append(f"\n*...and {len(transitions)-20} more transition events*")
        else:
            report_lines.append("\nNo ID transition events detected.")

        # Association Quality
        report_lines.append("\n## Association Quality Analysis")
        total_detections = self.summary.get('total_associations', 0)
        success_rate = self.summary.get('success_rate', 0)
        unassociated_rate = 100 - success_rate

        report_lines.append(f"- **Associated Detections**: {int(total_detections * success_rate / 100)} ({success_rate:.1f}%)")
        report_lines.append(f"- **Unassociated Detections**: {int(total_detections * unassociated_rate / 100)} ({unassociated_rate:.1f}%)")

        # Recommendations
        report_lines.append("\n## Recommendations")
        if success_rate < 50:
            report_lines.append("- ⚠️ **Low association rate** - Consider:")
            report_lines.append("  - Increasing proximity threshold (currently 200cm)")
            report_lines.append("  - Checking UWB tag synchronization")
            report_lines.append("  - Verifying homography calibration")
        elif success_rate < 75:
            report_lines.append("- ⚡ **Moderate association rate** - Consider:")
            report_lines.append("  - Fine-tuning proximity threshold")
            report_lines.append("  - Reviewing frames with low association")
        else:
            report_lines.append("- ✅ **Good association rate**")

        if len(transitions) > len(tag_stats) * 3:
            report_lines.append("- ⚠️ **High ID transition rate** - Players are frequently losing/regaining IDs")
            report_lines.append("  - Consider adjusting ByteTrack buffer (currently 50 frames)")
            report_lines.append("  - Increase fallback window (currently ±30 frames)")

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"✅ Statistics report saved: {output_path}")
