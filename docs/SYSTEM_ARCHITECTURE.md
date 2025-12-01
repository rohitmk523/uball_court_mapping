# System Architecture

## Table of Contents
1. [Overview](#overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Architecture Diagrams](#architecture-diagrams)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)

---

## Overview

The Basketball Court Tracking System is a full-stack web application that combines **Ultra-Wideband (UWB)** positioning data with **computer vision** to track basketball players on a court. The system provides real-time visualization, video-to-court calibration, and automated player-to-tag matching.

### Key Capabilities

- **UWB Tag Processing**: Parse raw log files containing millions of position records
- **Court Visualization**: Render basketball courts from CAD files (DXF format)
- **Video Analysis**: Detect and track players using YOLOv11 and ByteTrack
- **Spatial Calibration**: Map court coordinates to video coordinates via homography
- **Temporal Synchronization**: Align video frames with UWB timestamps
- **Real-time Animation**: Playback tag positions with adjustable FPS (1-30)

---

## System Components

### 1. Backend (FastAPI)

**Location**: `app/`

The backend is built with FastAPI, a modern Python web framework that provides:
- Automatic API documentation (Swagger UI)
- Async/await support for concurrent operations
- Type validation via Pydantic models
- High performance (comparable to NodeJS/Go)

#### Core Modules

**`app/main.py`** - Application Entry Point
```python
# Initializes FastAPI app
# Registers API routers
# Serves static files and templates
# Configures CORS if needed
```

**`app/core/config.py`** - Configuration Management
```python
# File paths (video, DXF, logs)
# Tracking parameters (thresholds, FPS limits)
# Directory structure management
# Constants and defaults
```

**`app/core/models.py`** - Data Models
```python
# Pydantic models for request/response validation
# TagPosition, Detection, Track, Calibration, etc.
# Type checking and automatic documentation
```

#### API Layer

**`app/api/`** - REST API Endpoints

- **`tags.py`**: UWB tag data operations
- **`court.py`**: Court geometry and rendering
- **`calibration.py`**: Homography calibration
- **`tracking.py`**: Video processing and matching

Each API module follows RESTful conventions:
- GET: Retrieve data
- POST: Create/process data
- DELETE: Remove data

#### Service Layer

**`app/services/`** - Business Logic

**`log_parser.py`** - UWB Log Processing
- Parses raw log files with regex
- Extracts tag ID, timestamp, X/Y coordinates
- Generates individual JSON files per tag
- Handles 1M+ records efficiently

**`dxf_parser.py`** - Court Rendering
- Loads DXF files using ezdxf library
- Extracts polylines, lines, circles
- Renders to PNG with proper scaling
- Handles coordinate system conversion

**`calibration.py`** - Homography Computation
- Computes 3x3 transformation matrix
- Uses RANSAC for outlier rejection
- Provides forward/inverse transformations
- Saves/loads calibration data

**`detector.py`** - Player Detection
- Wraps YOLOv11 models (nano to xlarge)
- Filters for person class (ID=0)
- Batch processes video frames
- Caches results to JSON

**`tracker.py`** - Multi-Object Tracking
- Implements ByteTrack algorithm
- Maintains track IDs across frames
- Handles occlusions and re-identification
- Fallback to simple position matching

**`matcher.py`** - Tag-to-Player Matching
- Transforms player positions to court coordinates
- Calculates Euclidean distances
- Matches within threshold (200cm default)
- Returns closest tag per player

### 2. Frontend (Vanilla JavaScript)

**Location**: `static/`

The frontend uses vanilla JavaScript (no frameworks) for simplicity and performance.

#### JavaScript Modules

**`static/js/utils.js`** - UI Utilities
```javascript
// Loading overlays and spinners
// Status messages and badges
// Input validation
// API error handling
// Number/timestamp formatting
```

**`static/js/courtRenderer.js`** - Court Visualization
```javascript
class CourtRenderer {
    // Draws court geometry on canvas
    // Handles coordinate transformations
    // Renders tags with golden angle colors
    // Scales to fit canvas dimensions
}
```

**`static/js/tagAnimator.js`** - Tag Animation
```javascript
class TagAnimator {
    // Loads tag data from API
    // Animates tags at specified FPS
    // Handles play/pause/reset
    // Timeline scrubbing
    // Frame interpolation
}
```

**`static/js/calibration.js`** - Calibration UI
```javascript
class CalibrationUI {
    // Dual-canvas point selection
    // Alternating court/video clicks
    // Visual markers and numbering
    // Undo functionality
    // Submits to API
}
```

**`static/js/tracking.js`** - Tracking Visualization
```javascript
class TrackingVisualizer {
    // Sync point management
    // Video processing trigger
    // Playback controls
    // Frame-by-frame navigation
    // Tag-to-player display
}
```

#### HTML Templates

**`templates/`** - Jinja2 Templates

- **`index.html`**: Tag visualization page
- **`calibration.html`**: Court-video calibration
- **`tracking.html`**: Player tracking and matching

Each template includes:
- Navigation bar for page switching
- Control panels for user input
- Canvas elements for visualization
- Status displays for feedback

#### CSS Styling

**`static/css/style.css`** - Unified Stylesheet

- Modern, clean design
- Responsive grid layouts
- Loading animations
- Interactive components
- Professional color scheme

### 3. Data Storage

**Location**: `data/`

The system uses file-based storage (no database) for simplicity:

```
data/
├── output/
│   └── tags/           # Individual tag JSON files
│       ├── 1586948.json
│       ├── 1587036.json
│       └── ...
├── calibration/
│   ├── court_image.png       # Rendered court
│   ├── calibration.json      # Homography matrix
│   └── sync_point.json       # Frame-timestamp sync
└── cache/
    └── detections/           # Cached YOLO results
        └── detections_0_500_5.json
```

**Advantages**:
- No database setup required
- Easy to inspect/debug
- Portable across systems
- Git-friendly (JSON format)

**Tag JSON Structure**:
```json
{
  "tag_id": 1587672,
  "positions": [
    {
      "timestamp": 301582,
      "x": 169.0,
      "y": 468.0,
      "datetime": "2025-11-22 02:26:45.184"
    }
  ]
}
```

**Calibration JSON Structure**:
```json
{
  "homography": [[...], [...], [...]],
  "court_points": [[x1,y1], [x2,y2], ...],
  "video_points": [[x1,y1], [x2,y2], ...],
  "timestamp": "2025-11-27T15:24:41.291528"
}
```

### 4. External Dependencies

**Computer Vision**:
- **OpenCV** (cv2): Image processing, homography computation
- **Ultralytics** (YOLOv11): Object detection
- **lapx**: Linear assignment for tracking

**Data Processing**:
- **NumPy**: Numerical operations, matrix math
- **ezdxf**: DXF file parsing

**Web Framework**:
- **FastAPI**: REST API server
- **Uvicorn**: ASGI server
- **Jinja2**: Template rendering

---

## Data Flow

### 1. UWB Tag Processing Flow

```
Raw Log File (session_1763778442483.log)
    ↓
[1] User clicks "Process Log File" button
    ↓
[2] POST /api/tags/process
    ↓
[3] log_parser.process_log()
    ↓
[4] Read file line by line
    ↓
[5] Regex: Extract tag_id, timestamp, x, y, datetime
    ↓
[6] Group positions by tag_id
    ↓
[7] Write data/output/tags/{tag_id}.json
    ↓
[8] Return: {status: "complete", tags_processed: 19}
    ↓
[9] Frontend loads tag list
    ↓
[10] GET /api/tags/list → [1586948, 1587036, ...]
    ↓
[11] For each tag: GET /api/tags/{tag_id}
    ↓
[12] TagAnimator stores all positions
    ↓
[13] Enable playback controls
```

**Processing Stats**:
- Input: 1.3M lines (raw log)
- Output: 19 JSON files (1,384,058 positions total)
- Time: ~2-5 seconds
- Timestamp range: 301564 to 268757500

### 2. Court Rendering Flow

```
DXF File (court_2.dxf)
    ↓
[1] GET /api/court/geometry
    ↓
[2] dxf_parser.get_court_geometry()
    ↓
[3] ezdxf.readfile("court_2.dxf")
    ↓
[4] Iterate modelspace entities
    ↓
[5] Extract LWPOLYLINE → polylines
    ↓
[6] Extract LINE → lines
    ↓
[7] Extract CIRCLE → circles (center, radius)
    ↓
[8] Calculate bounding box (min_x, min_y, max_x, max_y)
    ↓
[9] Return CourtGeometry object
    ↓
[10] GET /api/court/image (on first request)
    ↓
[11] dxf_parser.generate_court_image()
    ↓
[12] Create blank image (with margin)
    ↓
[13] Scale: 2 pixels per cm
    ↓
[14] Draw polylines, lines, circles
    ↓
[15] Flip Y-axis (court Y-up → image Y-down)
    ↓
[16] Save to data/calibration/court_image.png
    ↓
[17] Return PNG file (191KB)
    ↓
[18] Frontend displays on canvas
```

**Court Dimensions**:
- Real court: 2460.2cm × 1730.8cm (~24.6m × 17.3m)
- Image: 5720×4261 pixels (with 200cm margin)
- Entities: 6 polylines, 1 line, 4 circles

### 3. Calibration Flow

```
Court Image + Video Frame
    ↓
[1] User loads video frame (frame number input)
    ↓
[2] Frontend extracts frame from video file
    ↓
[3] Display court (left) and video (right)
    ↓
[4] User clicks point on court canvas
    ↓
[5] JavaScript records canvas coordinates
    ↓
[6] Transform to court coordinates (reverse scale/offset)
    ↓
[7] Highlight video canvas (user clicks next)
    ↓
[8] User clicks corresponding point on video
    ↓
[9] Record video pixel coordinates
    ↓
[10] Draw numbered markers on both canvases
    ↓
[11] Repeat steps 4-10 (minimum 10 points)
    ↓
[12] User clicks "Submit Calibration"
    ↓
[13] POST /api/calibration/points
    ↓
[14] Backend receives:
     - court_points: [[x1,y1], [x2,y2], ...]
     - video_points: [[x1,y1], [x2,y2], ...]
    ↓
[15] calibration.compute_homography()
    ↓
[16] cv2.findHomography(court_pts, video_pts, cv2.RANSAC, 5.0)
    ↓
[17] Returns 3×3 matrix H
    ↓
[18] Save to data/calibration/calibration.json
    ↓
[19] Return homography to frontend
    ↓
[20] Display "Calibration successful"
```

**Homography Matrix**:
```
H = | h11  h12  h13 |
    | h21  h22  h23 |
    | h31  h32  h33 |
```

**Transformation**:
- Court point (xc, yc) → Video point (xv, yv)
- Apply: [xv, yv, 1] = H × [xc, yc, 1]
- Normalize by h33

### 4. Video Processing Flow

```
Video File (GX020018.MP4)
    ↓
[1] User sets sync point
    ↓
[2] POST /api/tracking/sync
    {video_frame: 100, uwb_timestamp: 301564}
    ↓
[3] Save to data/calibration/sync_point.json
    ↓
[4] User configures processing:
    - Start frame: 0
    - End frame: 500
    - Frame skip: 5
    - Confidence: 0.5
    ↓
[5] POST /api/tracking/process
    ↓
[6] detector.process_video_batch()
    ↓
[7] cv2.VideoCapture(video_path)
    ↓
[8] For each frame in range (with skip):
    ↓
[9] cap.read() → frame image
    ↓
[10] model.predict(frame, conf=0.5, classes=[0])
    ↓
[11] Filter for person class (ID=0)
    ↓
[12] Extract bounding boxes [x1, y1, x2, y2]
    ↓
[13] Calculate bottom center: ((x1+x2)/2, y2)
    ↓
[14] Store Detection objects
    ↓
[15] Cache to JSON: detections_0_500_5.json
    ↓
[16] tracker.track_video(detections_per_frame)
    ↓
[17] ByteTrack assigns track IDs
    ↓
[18] Maintain IDs across frames (handle occlusions)
    ↓
[19] Return tracks_per_frame
    ↓
[20] For each frame: matcher.match_tags_to_players()
    ↓
[21] Load calibration matrix H
    ↓
[22] For each detected player:
    ↓
[23] Transform bbox_bottom_center (video → court)
    ↓
[24] Calculate timestamp for frame:
    timestamp = sync_uwb + (frame - sync_frame) × (1/FPS) × 1M
    ↓
[25] Load tags at this timestamp (tolerance: ±1M)
    ↓
[26] For each tag: distance = sqrt((xp-xt)² + (yp-yt)²)
    ↓
[27] Match to closest tag if distance < 200cm
    ↓
[28] Store matches
    ↓
[29] Return processing summary to frontend
    ↓
[30] Frontend enables playback
```

**Detection Cache**:
- Avoids reprocessing same frames
- Key: `detections_{start}_{end}_{skip}.json`
- Contains all detections for frame range

### 5. Playback Flow

```
Processed Video Data
    ↓
[1] User clicks "Play"
    ↓
[2] TrackingVisualizer.play()
    ↓
[3] Start animation loop (requestAnimationFrame)
    ↓
[4] Increment currentFrameIndex
    ↓
[5] Get frame ID from processedFrames[index]
    ↓
[6] GET /api/tracking/matched/{frame_id}
    ↓
[7] Backend calculates timestamp for frame
    ↓
[8] GET /api/tags/at/{timestamp}
    ↓
[9] Filter tags within tolerance (±1M)
    ↓
[10] Return matched tags with distances
    ↓
[11] Frontend updates court visualization
    ↓
[12] courtRenderer.updateTags(tags)
    ↓
[13] Clear canvas
    ↓
[14] Draw court geometry
    ↓
[15] For each tag: draw circle at (x, y)
    ↓
[16] Color by tag_id (golden angle: hue = id × 137.5° mod 360°)
    ↓
[17] Update statistics display:
    - Current frame / Total frames
    - UWB timestamp
    - Players detected
    - Tags matched
    ↓
[18] Loop to step 4 (~10 FPS)
```

---

## Architecture Diagrams

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Web Browser                          │
│  ┌──────────────┬──────────────┬────────────────────┐  │
│  │   Tag Viz    │ Calibration  │    Tracking        │  │
│  │   (index)    │              │                    │  │
│  └──────────────┴──────────────┴────────────────────┘  │
│           │              │                  │           │
│           │ API Calls (Fetch/JSON)         │           │
└───────────┼──────────────┼──────────────────┼───────────┘
            ↓              ↓                  ↓
┌───────────────────────────────────────────────────────────┐
│                  FastAPI Backend                          │
│  ┌──────────┬────────────┬──────────────┬──────────────┐ │
│  │ Tags API │ Court API  │ Calib API    │ Tracking API │ │
│  └────┬─────┴─────┬──────┴──────┬───────┴──────┬───────┘ │
│       │           │             │              │          │
│       ↓           ↓             ↓              ↓          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Service Layer                           │ │
│  │  ┌──────────┬──────────┬──────────┬──────────────┐  │ │
│  │  │LogParser │DXFParser │Calibrate │Detector      │  │ │
│  │  └──────────┴──────────┴──────────┴──────────────┘  │ │
│  │  ┌──────────┬──────────┐                            │ │
│  │  │ Tracker  │ Matcher  │                            │ │
│  │  └──────────┴──────────┘                            │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────┬───────────────────────────────┬─────────────┘
              ↓                               ↓
┌──────────────────────────┐    ┌──────────────────────────┐
│   File System (Data)     │    │  External Libraries      │
│  - Tag JSON files        │    │  - OpenCV (cv2)          │
│  - Court PNG             │    │  - YOLOv11 (ultralytics) │
│  - Calibration JSON      │    │  - ezdxf                 │
│  - Detection cache       │    │  - NumPy                 │
└──────────────────────────┘    └──────────────────────────┘
```

### Data Processing Pipeline

```
INPUT                PROCESSING               OUTPUT
─────                ──────────               ──────

UWB Log     ──────►  Log Parser    ──────►   Tag JSONs
(1.3M lines)         (Regex)                 (19 files)
                          │
                          └──────►  Time Range: 301564-268757500

DXF File    ──────►  DXF Parser    ──────►   Court PNG
(CAD)                (ezdxf)                 (5720×4261)
                          │
                          └──────►  Geometry: polylines, circles

Video +     ──────►  Calibration   ──────►   Homography
Court                (RANSAC)                Matrix (3×3)
(10-15 pts)               │
                          └──────►  Court ↔ Video mapping

Video       ──────►  YOLOv11       ──────►   Detections
(MP4)                (Person)                (Bounding boxes)
                          │
                          ↓
                     ByteTrack      ──────►   Tracks
                     (IDs)                    (Track IDs)
                          │
                          ↓
Tracks +    ──────►  Matcher       ──────►   Tag-Player
Tags +               (Distance)              Associations
Homography               │
                         └──────►  Within 200cm threshold
```

### Request-Response Flow (Example: Get Matched Tags)

```
Browser                FastAPI               Services              File System
───────                ───────               ────────              ───────────

GET /api/tracking/matched/150
   │
   ├──────────────────────►│
   │                       │
   │                       │ Load sync point
   │                       ├────────────────►│
   │                       │                 │ Read sync_point.json
   │                       │                 ├──────────────────►│
   │                       │                 │◄──────────────────┤
   │                       │◄────────────────┤
   │                       │ {frame:100, uwb:301564}
   │                       │
   │                       │ Calculate timestamp
   │                       │ ts = 301564 + (150-100)×(1/30)×1M
   │                       │ ts = 1,969,897
   │                       │
   │                       │ Get tags at timestamp
   │                       ├────────────────►│
   │                       │                 │ Read tag JSONs
   │                       │                 ├──────────────────►│
   │                       │                 │◄──────────────────┤
   │                       │◄────────────────┤
   │                       │ [tag1, tag2, ...]
   │                       │
   │                       │ Response JSON
   │◄──────────────────────┤
   │                       │
   │ {frame:150, timestamp:1969897, tags:[...]}
```

---

## Technology Stack

### Backend Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12 | Core language |
| FastAPI | 0.109.0 | Web framework |
| Uvicorn | 0.27.0 | ASGI server |
| Pydantic | 2.5.0 | Data validation |
| OpenCV | 4.9.0.80 | Computer vision |
| Ultralytics | 8.1.0 | YOLOv11 |
| ezdxf | 1.2.0 | DXF parsing |
| NumPy | 1.26.0 | Numerical ops |
| lapx | 0.9.2 | ByteTrack |

### Frontend Stack

| Technology | Purpose |
|-----------|---------|
| Vanilla JavaScript (ES6+) | UI logic |
| HTML5 Canvas | Graphics rendering |
| CSS3 | Styling & animations |
| Fetch API | HTTP requests |
| No frameworks | Simplicity & performance |

### Development Tools

| Tool | Purpose |
|------|---------|
| Conda | Environment management |
| Git | Version control |
| Chrome DevTools | Frontend debugging |
| Postman/curl | API testing |

---

## Design Patterns

### 1. Singleton Pattern

**Used for**: Detector and Tracker instances

**Why**: YOLO model loading is expensive (100-500MB), avoid repeated initialization.

```python
# detector.py
_detector_instance = None

def get_detector(model_name='yolo11n.pt', conf_threshold=0.5):
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlayerDetector(model_name, conf_threshold)
    return _detector_instance
```

### 2. Service Layer Pattern

**Separation of Concerns**:
- **API Layer** (`app/api/`): HTTP handling, request validation
- **Service Layer** (`app/services/`): Business logic, algorithms
- **Data Layer**: File I/O operations

**Benefits**:
- Easier testing (mock services)
- Clear responsibilities
- Reusable services

### 3. Repository Pattern (Simplified)

**File-based storage** acts as repository:
- Tags stored in `data/output/tags/{tag_id}.json`
- Calibration in `data/calibration/calibration.json`
- No database abstractions needed

### 4. Strategy Pattern

**Multiple tracking strategies**:
```python
# tracker.py
if byte_track_available:
    return self._byte_track(detections)
else:
    return self._simple_tracking(detections)
```

### 5. Observer Pattern (Frontend)

**Event-driven UI updates**:
```javascript
// Animation loop observes state changes
animate() {
    if (this.isPlaying) {
        this.updateFrame();
        requestAnimationFrame(() => this.animate());
    }
}
```

### 6. Factory Pattern

**Detection result creation**:
```python
# detector.py
def _create_detection(self, box, conf, class_id):
    return Detection(
        bbox=[x1, y1, x2, y2],
        confidence=conf,
        class_id=class_id,
        bbox_bottom_center=((x1+x2)/2, y2)
    )
```

---

## System Characteristics

### Scalability

**Current**: Single-user, single-video
**Limitations**:
- File-based storage (not concurrent-safe)
- In-memory processing
- No distributed processing

**Scale Potential**:
- Replace file storage with database (PostgreSQL)
- Add message queue for video processing (Celery + Redis)
- Horizontal scaling with load balancer

### Performance

**Processing Times**:
- Log parsing: 2-5 seconds (1.3M records)
- Court rendering: <1 second (first time)
- Calibration: <100ms (10-15 points)
- YOLO detection: ~2-5 min (500 frames, skip=5)

**Optimization**:
- Detection caching (avoid reprocessing)
- Lazy loading (load tags on demand)
- Frame skipping (configurable)

### Reliability

**Error Handling**:
- API: Try-catch with status codes
- Frontend: Loading indicators, error messages
- Validation: Input checks before processing

**Data Integrity**:
- JSON schema validation (Pydantic)
- Atomic file writes
- Backup calibration data

### Security

**Current State**:
- Local deployment only
- No authentication
- No encryption

**Production Considerations**:
- Add authentication (JWT tokens)
- HTTPS/TLS encryption
- Input sanitization
- Rate limiting

---

## Directory Structure

```
uball_court_mapping/
├── app/                        # Backend application
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── tags.py             # Tag data endpoints
│   │   ├── court.py            # Court geometry endpoints
│   │   ├── calibration.py      # Calibration endpoints
│   │   └── tracking.py         # Tracking/matching endpoints
│   ├── core/                   # Core configuration
│   │   ├── __init__.py
│   │   ├── config.py           # Settings and paths
│   │   └── models.py           # Pydantic models
│   └── services/               # Business logic
│       ├── __init__.py
│       ├── log_parser.py       # UWB log processing
│       ├── dxf_parser.py       # DXF court parsing
│       ├── calibration.py      # Homography computation
│       ├── detector.py         # YOLO wrapper
│       ├── tracker.py          # ByteTrack wrapper
│       └── matcher.py          # Tag-player matching
├── static/                     # Frontend assets
│   ├── css/
│   │   └── style.css           # Application styles
│   └── js/
│       ├── utils.js            # UI utilities
│       ├── courtRenderer.js    # Court visualization
│       ├── tagAnimator.js      # Tag animation
│       ├── calibration.js      # Calibration UI
│       └── tracking.js         # Tracking UI
├── templates/                  # HTML templates
│   ├── index.html              # Tag visualization page
│   ├── calibration.html        # Calibration page
│   └── tracking.html           # Tracking page
├── data/                       # Data storage
│   ├── output/
│   │   └── tags/               # Tag JSON files
│   ├── calibration/            # Calibration data
│   │   ├── court_image.png
│   │   ├── calibration.json
│   │   └── sync_point.json
│   └── cache/
│       └── detections/         # YOLO cache
├── docs/                       # Documentation
│   ├── SYSTEM_ARCHITECTURE.md  # This file
│   ├── API_REFERENCE.md
│   ├── USER_GUIDE.md
│   ├── TECHNICAL_DETAILS.md
│   └── FUTURE_IMPROVEMENTS.md
├── logs/                       # Input log files
│   └── session_1763778442483.log
├── dxf/                        # DXF court files
│   └── court_2.dxf
├── GX020018.MP4                # Video file (11GB)
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
├── start_server.sh             # Startup script
├── README.md                   # Quick start guide
└── TEST_REPORT.md              # Testing results
```

---

## Deployment Architecture

### Development Setup

```
Developer Machine
├── Conda Environment (court_tracking)
├── Uvicorn Server (localhost:8000)
├── Browser (UI)
└── File System (data/)
```

### Production Considerations

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (nginx)       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ FastAPI  │   │ FastAPI  │   │ FastAPI  │
        │ Worker 1 │   │ Worker 2 │   │ Worker 3 │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            ↓
                  ┌──────────────────┐
                  │   PostgreSQL     │
                  │   (Tag Data)     │
                  └──────────────────┘
                            ↓
                  ┌──────────────────┐
                  │   Redis          │
                  │   (Cache/Queue)  │
                  └──────────────────┘
                            ↓
                  ┌──────────────────┐
                  │   Object Storage │
                  │   (Videos/Images)│
                  └──────────────────┘
```

---

This architecture document provides a complete overview of the system design. For detailed usage instructions, see **USER_GUIDE.md**. For API specifications, see **API_REFERENCE.md**.
