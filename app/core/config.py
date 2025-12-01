"""Configuration settings for the application."""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TAGS_DIR = DATA_DIR / "tags"
CALIBRATION_DIR = DATA_DIR / "calibration"
CACHE_DIR = DATA_DIR / "cache"
DETECTIONS_CACHE_DIR = CACHE_DIR / "detections"

# Source data files
LOG_FILE = PROJECT_ROOT / "session_1763778442483.log"
DXF_FILE = PROJECT_ROOT / "court_2.dxf"
VIDEO_FILE = PROJECT_ROOT / "GX020018.MP4"

# Calibration files
COURT_IMAGE_FILE = CALIBRATION_DIR / "court_image.png"
VIDEO_FRAME_FILE = CALIBRATION_DIR / "video_frame.png"
HOMOGRAPHY_FILE = CALIBRATION_DIR / "homography.json"
SYNC_FILE = CALIBRATION_DIR / "sync.json"

# Tracking parameters
TAG_MATCH_THRESHOLD_CM = 200  # Distance threshold for tag-to-player matching
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # YOLO detection confidence
TRACK_BUFFER = 30  # ByteTrack buffer frames
MATCH_THRESH = 0.8  # ByteTrack matching threshold

# Visualization parameters
DEFAULT_FPS = 15
MAX_FPS = 30
MIN_FPS = 1

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [TAGS_DIR, CALIBRATION_DIR, CACHE_DIR, DETECTIONS_CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
