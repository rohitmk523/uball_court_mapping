"""
Data models for court video generation and event logging.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayerCourtPosition:
    """Represents a player's position projected onto the court."""
    detection_index: int  # Index in YOLO detections list
    bbox: List[int]  # [x1, y1, x2, y2] in video frame
    video_bottom_x: int  # Bottom-center X in video pixels
    video_bottom_y: int  # Bottom-center Y in video pixels
    court_x: float  # X coordinate on court (cm)
    court_y: float  # Y coordinate on court (cm)
    canvas_x: int  # X pixel on court canvas
    canvas_y: int  # Y pixel on court canvas
    confidence: float  # YOLO confidence score

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class UWBTagPosition:
    """Represents a UWB tag's position on the court."""
    tag_id: int
    x_uwb: float  # X in UWB coordinate system (cm)
    y_uwb: float  # Y in UWB coordinate system (cm)
    court_x: float  # X in calibration coordinate system (cm)
    court_y: float  # Y in calibration coordinate system (cm)
    canvas_x: int  # X pixel on court canvas
    canvas_y: int  # Y pixel on court canvas
    datetime_str: str  # ISO datetime string

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TagEvent:
    """Represents a tagging event when a player is within range of a tag."""
    timestamp: float  # Video time in seconds
    frame_number: int
    player_index: int  # Detection index in frame
    player_bbox: List[int]  # [x1, y1, x2, y2] in video
    player_court_x: float  # cm
    player_court_y: float  # cm
    tag_id: int
    tag_x: float  # UWB cm
    tag_y: float  # UWB cm
    distance_cm: float
    datetime_utc: str  # ISO format

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProcessingMetadata:
    """Metadata about the video processing run."""
    video_file: str
    sync_offset_seconds: float
    tagging_threshold_cm: float
    processing_date: str  # ISO format
    yolo_model: str
    yolo_confidence: float
    total_frames_processed: int
    duration_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProcessingSummary:
    """Summary statistics from the video processing."""
    total_events: int
    unique_players: int
    unique_tags: int
    duration_seconds: float
    frames_with_players: int
    frames_with_tags: int
    frames_with_tagging: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProcessingResult:
    """Result of a court video processing operation."""
    success: bool
    output_video_path: Optional[str] = None
    output_events_path: Optional[str] = None
    events_count: int = 0
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TagEventLogger:
    """Handles logging and exporting of tag events."""

    def __init__(self):
        """Initialize the event logger."""
        self.events: List[TagEvent] = []
        self.metadata: Optional[ProcessingMetadata] = None

    def set_metadata(self, metadata: ProcessingMetadata):
        """Set processing metadata."""
        self.metadata = metadata

    def add_event(self, event: TagEvent):
        """Add a tagging event."""
        self.events.append(event)

    def clear_events(self):
        """Clear all logged events."""
        self.events.clear()

    def get_summary(self) -> ProcessingSummary:
        """Calculate summary statistics from logged events."""
        if not self.events:
            return ProcessingSummary(
                total_events=0,
                unique_players=0,
                unique_tags=0,
                duration_seconds=0.0,
                frames_with_players=0,
                frames_with_tags=0,
                frames_with_tagging=0
            )

        # Calculate statistics
        unique_players = len(set((e.frame_number, e.player_index) for e in self.events))
        unique_tags = len(set(e.tag_id for e in self.events))
        unique_frames = len(set(e.frame_number for e in self.events))

        # Duration from first to last event
        timestamps = [e.timestamp for e in self.events]
        duration = max(timestamps) - min(timestamps) if timestamps else 0.0

        return ProcessingSummary(
            total_events=len(self.events),
            unique_players=unique_players,
            unique_tags=unique_tags,
            duration_seconds=duration,
            frames_with_players=unique_players,  # Approximation
            frames_with_tags=len(set(e.tag_id for e in self.events)),
            frames_with_tagging=unique_frames
        )

    def export_to_json(self, output_path: str) -> bool:
        """
        Export events to JSON file.

        Args:
            output_path: Path to output JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if needed
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Build export structure
            export_data = {
                "metadata": self.metadata.to_dict() if self.metadata else {},
                "summary": self.get_summary().to_dict(),
                "events": [event.to_dict() for event in self.events]
            }

            # Write to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported {len(self.events)} events to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export events to JSON: {e}")
            return False

    def get_events_by_tag(self, tag_id: int) -> List[TagEvent]:
        """Get all events for a specific tag."""
        return [e for e in self.events if e.tag_id == tag_id]

    def get_events_by_frame(self, frame_number: int) -> List[TagEvent]:
        """Get all events in a specific frame."""
        return [e for e in self.events if e.frame_number == frame_number]

    def get_events_in_timerange(self, start_time: float, end_time: float) -> List[TagEvent]:
        """Get all events within a time range."""
        return [e for e in self.events
                if start_time <= e.timestamp <= end_time]
