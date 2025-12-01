"""Pydantic models for API request/response validation."""
from typing import List, Optional, Tuple
from pydantic import BaseModel


class TagPosition(BaseModel):
    """Single tag position at a specific timestamp."""
    timestamp: int  # UWB timestamp
    x: float  # X coordinate in cm
    y: float  # Y coordinate in cm
    datetime: str  # ISO datetime string


class TagData(BaseModel):
    """Complete data for a single tag."""
    tag_id: int
    positions: List[TagPosition]


class CourtBounds(BaseModel):
    """Bounding box for the court."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width: float
    height: float


class CourtGeometry(BaseModel):
    """Court geometry data."""
    polylines: List[List[Tuple[float, float]]]
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    circles: List[Tuple[Tuple[float, float], float]]  # (center, radius)
    arcs: List[Tuple[Tuple[float, float], float, float, float]] = []  # (center, radius, start_angle, end_angle)
    bounds: CourtBounds


class CalibrationPoints(BaseModel):
    """Calibration correspondence points."""
    court_points: List[List[float]]  # [[x1,y1], [x2,y2], ...]
    video_points: List[List[float]]  # [[x1,y1], [x2,y2], ...]


class Calibration(BaseModel):
    """Saved calibration data."""
    homography: List[List[float]]  # 3x3 matrix
    court_points: List[List[float]]
    video_points: List[List[float]]
    timestamp: str


class SyncPoint(BaseModel):
    """Video-to-UWB timestamp synchronization."""
    video_frame: int
    uwb_timestamp: int


class Detection(BaseModel):
    """YOLO detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    bbox_bottom_center: Tuple[float, float]


class Track(BaseModel):
    """Tracked object."""
    track_id: int
    bbox: List[float]
    confidence: float
    bbox_bottom_center: Tuple[float, float]


class TagMatch(BaseModel):
    """Tag matched to a tracked player."""
    track_id: int
    tag_id: Optional[int]
    distance_cm: Optional[float]


class FrameResult(BaseModel):
    """Complete result for a single frame."""
    frame_id: int
    timestamp: Optional[int]  # UWB timestamp if synced
    tracks: List[Track]
    tags: List[TagPosition]
    matches: List[TagMatch]


class ProcessingStatus(BaseModel):
    """Status of a processing operation."""
    status: str  # "processing", "complete", "error"
    progress: Optional[float]  # 0.0 to 1.0
    message: Optional[str]
    result: Optional[dict]
