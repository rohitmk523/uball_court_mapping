"""
Automatic synchronization between video and UWB log timestamps using UTC.
"""
import subprocess
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Synchronizer:
    """Handles automatic synchronization between video and UWB log data using UTC timestamps."""

    def __init__(self):
        self.video_start_utc = None
        self.uwb_start_utc = None
        self.sync_offset_seconds = None

    def get_video_start_time(self, video_path: str) -> Optional[datetime]:
        """
        Extract video creation time from metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Video creation time as datetime object, or None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Extract creation_time from format tags
            creation_time_str = data['format']['tags'].get('creation_time')

            if not creation_time_str:
                logger.error("No creation_time found in video metadata")
                return None

            # Parse ISO 8601 format: 2025-11-22T02:27:30.000000Z
            video_start = datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))

            logger.info(f"Video start time: {video_start} UTC")
            return video_start

        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract video start time: {e}")
            return None

    def get_uwb_start_time(self, log_file: str) -> Optional[datetime]:
        """
        Extract first UWB timestamp from log file.

        Args:
            log_file: Path to UWB log file

        Returns:
            First log timestamp as datetime object, or None if failed
        """
        try:
            log_path = Path(log_file)
            if not log_path.exists():
                logger.error(f"Log file not found: {log_file}")
                return None

            with open(log_path, 'r') as f:
                first_line = f.readline().strip()

            if not first_line:
                logger.error("Log file is empty")
                return None

            # Parse format: "2025-11-22 02:26:40.578 | Tag ..."
            timestamp_str = first_line.split('|')[0].strip()

            # Parse: YYYY-MM-DD HH:MM:SS.mmm
            uwb_start = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')

            logger.info(f"UWB log start time: {uwb_start} (local/UTC)")
            return uwb_start

        except Exception as e:
            logger.error(f"Failed to extract UWB start time: {e}")
            return None

    def calculate_sync_offset(
        self,
        video_path: str,
        log_file: str
    ) -> Optional[float]:
        """
        Calculate synchronization offset between video and UWB log.

        The offset represents how many seconds into the UWB log timeline
        the video starts.

        Args:
            video_path: Path to video file
            log_file: Path to UWB log file

        Returns:
            Sync offset in seconds (positive = video started after log),
            or None if calculation failed
        """
        try:
            # Extract timestamps
            self.video_start_utc = self.get_video_start_time(video_path)
            self.uwb_start_utc = self.get_uwb_start_time(log_file)

            if self.video_start_utc is None or self.uwb_start_utc is None:
                logger.error("Failed to extract timestamps for sync calculation")
                return None

            # Make uwb_start timezone-aware to match video_start
            # Assuming UWB log timestamps are in UTC
            from datetime import timezone
            uwb_start_aware = self.uwb_start_utc.replace(tzinfo=timezone.utc)

            # Calculate offset: (video_start - uwb_start) in seconds
            time_diff = self.video_start_utc - uwb_start_aware
            self.sync_offset_seconds = time_diff.total_seconds()

            logger.info(f"Calculated sync offset: {self.sync_offset_seconds:.3f} seconds")
            logger.info(f"  Video started {self.sync_offset_seconds:.3f}s after UWB log")

            return self.sync_offset_seconds

        except Exception as e:
            logger.error(f"Sync offset calculation failed: {e}")
            return None

    def get_sync_offset(
        self,
        video_path: str,
        log_file: str,
        cache: bool = True
    ) -> float:
        """
        Get synchronization offset, calculating if not already cached.

        Args:
            video_path: Path to video file
            log_file: Path to UWB log file
            cache: Use cached offset if already calculated

        Returns:
            Sync offset in seconds

        Raises:
            ValueError: If sync offset calculation fails
        """
        if cache and self.sync_offset_seconds is not None:
            return self.sync_offset_seconds

        offset = self.calculate_sync_offset(video_path, log_file)

        if offset is None:
            raise ValueError("Failed to calculate sync offset")

        return offset

    def save_sync_data(self, output_file: str) -> bool:
        """
        Save synchronization data to JSON file.

        Args:
            output_file: Path to output JSON file

        Returns:
            True if successful
        """
        try:
            if self.sync_offset_seconds is None:
                logger.error("No sync data to save")
                return False

            data = {
                "video_start_utc": self.video_start_utc.isoformat() if self.video_start_utc else None,
                "uwb_start_utc": self.uwb_start_utc.isoformat() if self.uwb_start_utc else None,
                "sync_offset_seconds": self.sync_offset_seconds
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved sync data to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save sync data: {e}")
            return False
