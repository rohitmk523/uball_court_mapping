# Basketball Court Tracking System

A comprehensive web-based system for tracking basketball players using UWB (Ultra-Wideband) tag positioning, video analysis with YOLOv11, and court-to-video calibration.

## Features

- **UWB Tag Visualization**: Real-time visualization of player positions from UWB tag data
- **Court Rendering**: Automatic court geometry extraction from DXF files
- **Video Calibration**: Manual correspondence point selection for homography computation
- **Player Detection**: YOLOv11-based player detection in video frames
- **SAM2 Segmentation** (Optional): Enhanced player segmentation for improved tracking accuracy
- **Multi-Object Tracking**: ByteTrack algorithm for consistent player tracking with optional mask-based features
- **Tag-to-Player Matching**: Automatic matching of UWB tags to detected players
- **Timestamp Synchronization**: Manual sync point setting for video-UWB alignment
- **Interactive Playback**: Frame-by-frame playback with FPS control (1-30 FPS)

## System Requirements

- Python 3.12
- Conda package manager
- 11GB+ storage (for video files)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation

### 1. Create Conda Environment

```bash
conda create -n court_tracking python=3.12 -y
conda activate court_tracking
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- FastAPI 0.109.0
- Ultralytics 8.1.0 (YOLOv11)
- OpenCV 4.9.0.80
- ezdxf 1.2.0
- lapx 0.9.2 (ByteTrack)

### 3. Verify Installation

```bash
python -c "import cv2, ultralytics, ezdxf; print('All dependencies installed successfully')"
```

### 4. Optional: Install SAM2 for Enhanced Segmentation

For improved tracking accuracy with precise player segmentation:

```bash
# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download SAM2 checkpoint
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

See [SAM2_INTEGRATION.md](SAM2_INTEGRATION.md) for detailed setup and usage instructions.

## Quick Start

### Starting the Server

```bash
./start_server.sh
```

Or manually:

```bash
conda activate court_tracking
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on **http://localhost:8000**

## Complete Workflow

### 1. Tag Visualization (http://localhost:8000/)

**Purpose**: View and animate UWB tag data

**Steps:**
1. Click "Process Log File" to parse UWB data
2. Use playback controls (Play/Pause/Reset)
3. Adjust FPS (1-30) for animation speed
4. Scrub timeline to jump to timestamps

**Data**: 19 tags, 1.3M positions, timestamp range 301564-268757500

### 2. Calibration (http://localhost:8000/calibration)

**Purpose**: Map court coordinates to video coordinates

**Steps:**
1. Load a video frame (enter frame number, click "Load Video Frame")
2. Click corresponding points on Court (left) and Video (right)
3. Select 10-15 well-distributed points (court line intersections work best)
4. Click "Submit Calibration"

**Tips:**
- Alternate clicks: Court → Video → Court → Video
- Use clear court markings (lines, corners, center circle)
- More points = better accuracy (10-15 recommended)

### 3. Tracking (http://localhost:8000/tracking)

**Purpose**: Process video with YOLO and match tags to players

**Step 3.1: Set Synchronization Point**
- Find an event visible in both video and UWB data
- Enter Video Frame and UWB Timestamp
- Click "Set Sync Point"

**Step 3.2: Process Video**
- Start Frame: 0 (or your start frame)
- End Frame: 500 (for testing, use smaller range)
- Frame Skip: 5 (process every 5th frame)
- Click "Process Video with YOLO"

**Processing Time**: ~2-5 minutes for 500 frames with skip=5

**Step 3.3: Playback**
- Use Play/Pause/Reset controls
- View statistics (frame count, timestamp, players, tags)
- Court panel shows UWB tag positions at current frame's timestamp

## API Endpoints

### Court
- `GET /api/court/geometry` - Court geometry
- `GET /api/court/image` - Rendered court image

### Tags
- `GET /api/tags/list` - All tag IDs
- `GET /api/tags/{tag_id}` - Tag position history
- `POST /api/tags/process` - Process UWB log file

### Calibration
- `GET /api/calibration/status` - Calibration status
- `POST /api/calibration/points` - Submit points & compute homography
- `DELETE /api/calibration/delete` - Clear calibration

### Tracking
- `GET /api/tracking/sync` - Get sync point
- `POST /api/tracking/sync` - Set sync point
- `POST /api/tracking/process` - Process video with YOLO
- `GET /api/tracking/matched/{frame}` - Get matched tags for frame

## Configuration

Edit `app/core/config.py`:

```python
# File paths
VIDEO_FILE = PROJECT_ROOT / "GX020018.MP4"
DXF_FILE = PROJECT_ROOT / "dxf" / "court_2.dxf"
LOG_FILE = PROJECT_ROOT / "logs" / "session_1763778442483.log"

# Tracking parameters
TAG_MATCH_THRESHOLD_CM = 200  # Tag-to-player distance threshold
DEFAULT_FPS = 15               # Animation FPS
MAX_FPS = 30                   # Maximum FPS
```

## Coordinate Systems

- **Court**: Origin (0,0) bottom-left, units in cm, Y-axis UP, range 2460x1730cm
- **Video**: Origin (0,0) top-left, units in pixels, Y-axis DOWN
- **Transformation**: Homography matrix (court ↔ video)

## Performance Tips

### Frame Skip Strategy

| Frame Skip | Speed | Resolution | Use Case |
|-----------|-------|------------|----------|
| 1 | Slow | Best | Final analysis |
| 3-5 | Moderate | Good | General tracking |
| 10 | Fast | Adequate | Testing |
| 30 | Very fast | Low | Quick setup |

### Optimization
- Test with small frame ranges first (100-200 frames)
- Use detection cache (auto-enabled)
- Adjust confidence threshold if needed
- Clear cache to force reprocessing: `rm -rf data/cache/detections/*`

## Troubleshooting

**Server Won't Start**
```bash
# Check port 8000
lsof -i :8000
# Kill if needed
kill -9 <PID>
```

**"Calibration Required" Error**
- Complete calibration workflow first
- Verify `data/calibration/calibration.json` exists

**"Sync Point Required" Error**
- Set sync point on tracking page first
- Verify `data/calibration/sync_point.json` exists

**YOLO Processing Fails**
```bash
# Test YOLO
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('OK')"
```

**Memory Issues**
- Use larger frame skip (5-10)
- Process smaller chunks (500 frames at a time)
- Clear cache: `rm -rf data/cache/*`

## Project Structure

```
uball_court_mapping/
├── app/
│   ├── api/          # API endpoints
│   ├── core/         # Config & models
│   └── services/     # Business logic
├── static/
│   ├── css/          # Styles (with loading indicators)
│   └── js/           # UI logic (utils.js, tracking.js, etc.)
├── templates/        # HTML pages
├── data/
│   ├── output/       # Tag JSON files
│   ├── calibration/  # Calibration data & court image
│   └── cache/        # YOLO detection cache
├── logs/             # UWB log files
├── dxf/              # Court DXF files
└── GX020018.MP4      # Video file (11GB)
```

## Camera Setup

- **Position**: Mounted on basketball hoop
- **View**: Far hoop and half court
- **Orientation**: Left court = Top video

## Technical Details

### Homography Computation
- Algorithm: `cv2.findHomography()` with RANSAC
- Threshold: 5.0 pixels
- Minimum points: 10

### YOLO Detection
- Model: YOLOv11n (nano)
- Person class ID: 0
- Confidence: 0.5 (default)

### ByteTrack Tracking
- Track buffer: 30 frames
- IOU threshold: 0.8
- Fallback: Position-based (100px)

### Tag Matching
- Distance metric: Euclidean in court coords
- Threshold: 200cm (2 meters)

## Test Report

See `TEST_REPORT.md` for comprehensive testing results.

## Development

### New Features
- **API Endpoint**: Add to `app/api/`
- **Service**: Add to `app/services/`
- **Frontend**: Add to `static/js/`

### Code Structure
- Backend: FastAPI with async/await
- Frontend: Vanilla JavaScript
- Styling: Pure CSS with animations
- Models: Pydantic validation
- Services: Singleton pattern

## Recent Improvements

- ✓ Loading overlays for long operations
- ✓ Progress indicators and spinners
- ✓ Input validation with error messages
- ✓ Enhanced status messages with icons
- ✓ Utility functions for consistent UX
- ✓ Timeout handling for API requests
- ✓ Better error handling throughout

## License

Internal use - Cellstrat

---

**Last Updated**: November 27, 2025
**Version**: 1.0.0
**Status**: Production Ready

For detailed test results, see `TEST_REPORT.md`
