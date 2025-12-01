# API Reference

Complete REST API documentation for the Basketball Court Tracking System.

## Table of Contents

1. [API Overview](#api-overview)
2. [Tags API](#tags-api)
3. [Court API](#court-api)
4. [Calibration API](#calibration-api)
5. [Tracking API](#tracking-api)
6. [Error Handling](#error-handling)
7. [Data Models](#data-models)

---

## API Overview

### Base URL

```
http://localhost:8000
```

### Response Format

All endpoints return JSON unless specified otherwise (e.g., image files).

### Common HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Resource not found |
| 500 | Internal Server Error | Server error |

### Interactive Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Tags API

Endpoints for managing UWB tag data.

### 1. Process Log File

Parse raw UWB log file and generate individual tag JSON files.

**Endpoint**: `POST /api/tags/process`

**Description**: Processes the UWB log file and creates individual JSON files for each tag.

**Request**: No body required

**Response**:
```json
{
  "status": "complete",
  "message": "Successfully processed 19 tags with 1384058 total positions",
  "tags_processed": 19,
  "total_positions": 1384058,
  "time_range": {
    "min": 301564,
    "max": 268757500
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/api/tags/process
```

**Processing Details**:
- Reads `logs/session_1763778442483.log`
- Extracts tag ID, timestamp, X/Y coordinates, datetime
- Groups positions by tag ID
- Writes to `data/output/tags/{tag_id}.json`
- Takes 2-5 seconds for ~1.3M records

**Log Format**:
```
2025-11-22 02:26:40.578 | Tag 1587672 | X=454 | Y=469 | Timestamp=268737020
```

---

### 2. Get Processing Status

Check if log processing is complete.

**Endpoint**: `GET /api/tags/status`

**Description**: Returns the current status of log processing.

**Response**:
```json
{
  "processed": true,
  "tag_count": 19,
  "message": "19 tag files found"
}
```

**Example**:
```bash
curl http://localhost:8000/api/tags/status
```

---

### 3. List All Tags

Get list of all available tag IDs.

**Endpoint**: `GET /api/tags/list`

**Description**: Returns array of all tag IDs that have been processed.

**Response**:
```json
[
  1586948,
  1587036,
  1587048,
  1587240,
  1587248,
  1587365,
  1587366,
  1587367,
  1587369,
  1587377,
  1587380,
  1587392,
  1587399,
  1587479,
  1587505,
  1587552,
  1587578,
  1587672,
  1587823
]
```

**Example**:
```bash
curl http://localhost:8000/api/tags/list
```

---

### 4. Get Tag Data

Retrieve complete position history for a specific tag.

**Endpoint**: `GET /api/tags/{tag_id}`

**Parameters**:
- `tag_id` (path, required): Tag ID (integer)

**Response**:
```json
{
  "tag_id": 1587672,
  "positions": [
    {
      "timestamp": 301582,
      "x": 169.0,
      "y": 468.0,
      "datetime": "2025-11-22 02:26:45.184"
    },
    {
      "timestamp": 301582,
      "x": 165.0,
      "y": 470.0,
      "datetime": "2025-11-22 02:26:45.250"
    }
  ]
}
```

**Position Object**:
- `timestamp`: UWB timestamp (microseconds)
- `x`: X coordinate in centimeters
- `y`: Y coordinate in centimeters
- `datetime`: ISO 8601 datetime string

**Example**:
```bash
curl http://localhost:8000/api/tags/1587672
```

**Error Response** (404):
```json
{
  "detail": "Tag 9999999 not found"
}
```

---

### 5. Get Tags at Timestamp

Get all tag positions at a specific timestamp.

**Endpoint**: `GET /api/tags/at/{timestamp}`

**Parameters**:
- `timestamp` (path, required): UWB timestamp (integer)
- `tolerance` (query, optional): Time tolerance in microseconds (default: 1000000)

**Description**: Returns positions of all tags within the tolerance window around the specified timestamp.

**Response**:
```json
{
  "timestamp": 301564,
  "tolerance": 1000000,
  "tags": [
    {
      "tag_id": 1587672,
      "x": 169.0,
      "y": 468.0,
      "timestamp": 301582,
      "datetime": "2025-11-22 02:26:45.184"
    },
    {
      "tag_id": 1587036,
      "x": 523.0,
      "y": 892.0,
      "timestamp": 301590,
      "datetime": "2025-11-22 02:26:45.192"
    }
  ]
}
```

**Example**:
```bash
# Default tolerance (1 second = 1,000,000 microseconds)
curl http://localhost:8000/api/tags/at/301564

# Custom tolerance (500ms)
curl http://localhost:8000/api/tags/at/301564?tolerance=500000
```

**Use Case**: Called during playback to get all visible tags at the current frame's timestamp.

---

### 6. Get Tags in Time Range

Get all tag positions within a time range.

**Endpoint**: `GET /api/tags/timerange/`

**Query Parameters**:
- `start` (required): Start timestamp (integer)
- `end` (required): End timestamp (integer)
- `tag_ids` (optional): Comma-separated tag IDs to filter

**Response**:
```json
{
  "start": 301564,
  "end": 400000,
  "tags": {
    "1587672": [
      {
        "timestamp": 301582,
        "x": 169.0,
        "y": 468.0,
        "datetime": "2025-11-22 02:26:45.184"
      }
    ],
    "1587036": [
      {
        "timestamp": 301590,
        "x": 523.0,
        "y": 892.0,
        "datetime": "2025-11-22 02:26:45.192"
      }
    ]
  }
}
```

**Example**:
```bash
# All tags in range
curl "http://localhost:8000/api/tags/timerange/?start=301564&end=400000"

# Specific tags only
curl "http://localhost:8000/api/tags/timerange/?start=301564&end=400000&tag_ids=1587672,1587036"
```

---

## Court API

Endpoints for court geometry and visualization.

### 1. Get Court Geometry

Retrieve parsed court geometry from DXF file.

**Endpoint**: `GET /api/court/geometry`

**Description**: Returns court structure including polylines, lines, and circles.

**Response**:
```json
{
  "type": "CourtGeometry",
  "bounds": {
    "min_x": 0.0,
    "min_y": 0.0,
    "max_x": 2460.2,
    "max_y": 1730.8,
    "width": 2460.2,
    "height": 1730.8
  },
  "polylines": [
    [
      [0.0, 0.0],
      [2460.2, 0.0],
      [2460.2, 1730.8],
      [0.0, 1730.8]
    ],
    [
      [157.5, 151.8],
      [2301.2, 151.8],
      [2301.2, 1578.2],
      [157.5, 1578.2]
    ]
  ],
  "lines": [
    [
      [1229.4, 151.8],
      [1229.4, 1578.2]
    ]
  ],
  "circles": [
    {
      "center": [1229.4, 865.1],
      "radius": 182.9
    },
    {
      "center": [1229.4, 865.1],
      "radius": 61.0
    }
  ]
}
```

**Coordinate System**:
- Origin: (0, 0) at bottom-left
- Units: Centimeters
- Y-axis: Points UP

**Example**:
```bash
curl http://localhost:8000/api/court/geometry
```

**Use Case**: Frontend loads this to render the court on canvas.

---

### 2. Get Court Bounds

Get just the bounding box of the court.

**Endpoint**: `GET /api/court/bounds`

**Response**:
```json
{
  "min_x": 0.0,
  "min_y": 0.0,
  "max_x": 2460.2,
  "max_y": 1730.8,
  "width": 2460.2,
  "height": 1730.8
}
```

**Example**:
```bash
curl http://localhost:8000/api/court/bounds
```

---

### 3. Get Court Image

Retrieve rendered court image as PNG.

**Endpoint**: `GET /api/court/image`

**Description**: Returns a pre-rendered PNG image of the court. Generates on first request if not exists.

**Response**: PNG image file (binary)

**Headers**:
- `Content-Type: image/png`
- `Content-Disposition: inline; filename="court.png"`

**Image Details**:
- Size: 5720×4261 pixels
- File size: ~191KB
- Scale: 2 pixels per cm
- Margin: 200cm around court

**Example**:
```bash
# View in browser
open http://localhost:8000/api/court/image

# Download
curl http://localhost:8000/api/court/image -o court.png
```

**Use Case**: Displayed in calibration UI for point selection.

---

### 4. Regenerate Court Image

Force regeneration of court image with custom parameters.

**Endpoint**: `POST /api/court/regenerate-image`

**Query Parameters**:
- `margin` (optional): Margin around court in cm (default: 200)
- `scale` (optional): Pixels per cm (default: 2.0)

**Response**:
```json
{
  "status": "success",
  "image_path": "/path/to/data/calibration/court_image.png",
  "message": "Court image regenerated with margin=200cm, scale=2px/cm"
}
```

**Example**:
```bash
# Default parameters
curl -X POST http://localhost:8000/api/court/regenerate-image

# Custom parameters
curl -X POST "http://localhost:8000/api/court/regenerate-image?margin=300&scale=3"
```

**Use Case**: If you want higher resolution or more margin for tags outside court.

---

## Calibration API

Endpoints for court-to-video homography calibration.

### 1. Get Calibration Status

Check if calibration has been performed.

**Endpoint**: `GET /api/calibration/status`

**Response** (Not calibrated):
```json
{
  "calibrated": false,
  "timestamp": null,
  "num_points": 0
}
```

**Response** (Calibrated):
```json
{
  "calibrated": true,
  "timestamp": "2025-11-27T15:24:41.291528",
  "num_points": 11
}
```

**Example**:
```bash
curl http://localhost:8000/api/calibration/status
```

---

### 2. Submit Calibration Points

Compute homography matrix from correspondence points.

**Endpoint**: `POST /api/calibration/points`

**Request Body**:
```json
{
  "court_points": [
    [300.0, 300.0],
    [2160.0, 300.0],
    [2160.0, 1430.0],
    [300.0, 1430.0],
    [1230.0, 865.0],
    [752.0, 865.0],
    [1707.0, 865.0],
    [1230.0, 152.0],
    [1230.0, 1578.0],
    [158.0, 616.0],
    [2301.0, 616.0]
  ],
  "video_points": [
    [100.0, 500.0],
    [1820.0, 500.0],
    [1820.0, 580.0],
    [100.0, 580.0],
    [960.0, 540.0],
    [480.0, 540.0],
    [1440.0, 540.0],
    [960.0, 500.0],
    [960.0, 580.0],
    [100.0, 520.0],
    [1820.0, 520.0]
  ]
}
```

**Requirements**:
- Minimum 10 points
- Arrays must have same length
- Points must be in same order (point i in court corresponds to point i in video)

**Response**:
```json
{
  "homography": [
    [0.5351101083044544, -0.5182271041153701, 301.44669549435247],
    [-0.0001291021980772559, -0.23969439152554647, 494.97523559476815],
    [-2.4583574005533623e-07, -0.0005400727379898777, 0.9999999999999999]
  ],
  "court_points": [[300.0, 300.0], ...],
  "video_points": [[100.0, 500.0], ...],
  "timestamp": "2025-11-27T15:24:41.291528"
}
```

**Homography Matrix**:
- 3×3 matrix for perspective transformation
- Computed using `cv2.findHomography()` with RANSAC
- RANSAC threshold: 5.0 pixels
- Saved to `data/calibration/calibration.json`

**Example**:
```bash
curl -X POST http://localhost:8000/api/calibration/points \
  -H "Content-Type: application/json" \
  -d '{
    "court_points": [[300, 300], [2160, 300], ...],
    "video_points": [[100, 500], [1820, 500], ...]
  }'
```

**Error Response** (400):
```json
{
  "detail": "At least 10 point pairs required for calibration"
}
```

---

### 3. Get Homography Matrix

Retrieve the current calibration matrix.

**Endpoint**: `GET /api/calibration/matrix`

**Response**:
```json
{
  "homography": [
    [0.535, -0.518, 301.446],
    [-0.0001, -0.239, 494.975],
    [-2.458e-07, -0.0005, 1.0]
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/api/calibration/matrix
```

**Error Response** (400):
```json
{
  "detail": "No calibration found. Please calibrate first."
}
```

---

### 4. Delete Calibration

Remove saved calibration data.

**Endpoint**: `DELETE /api/calibration/delete`

**Response**:
```json
{
  "status": "success",
  "message": "Calibration deleted"
}
```

**Example**:
```bash
curl -X DELETE http://localhost:8000/api/calibration/delete
```

**Effect**: Deletes `data/calibration/calibration.json`

---

## Tracking API

Endpoints for video processing, player tracking, and tag matching.

### 1. Get Sync Point

Retrieve the current video-UWB synchronization point.

**Endpoint**: `GET /api/tracking/sync`

**Response** (Set):
```json
{
  "video_frame": 100,
  "uwb_timestamp": 301564
}
```

**Response** (Not set):
```json
null
```

**Example**:
```bash
curl http://localhost:8000/api/tracking/sync
```

---

### 2. Set Sync Point

Define synchronization between video frame and UWB timestamp.

**Endpoint**: `POST /api/tracking/sync`

**Request Body**:
```json
{
  "video_frame": 100,
  "uwb_timestamp": 301564
}
```

**Parameters**:
- `video_frame`: Frame number in video (0-indexed)
- `uwb_timestamp`: Corresponding UWB timestamp (microseconds)

**Response**:
```json
{
  "status": "success",
  "message": "Sync point set: frame 100 = timestamp 301564",
  "sync": {
    "video_frame": 100,
    "uwb_timestamp": 301564
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/api/tracking/sync \
  -H "Content-Type: application/json" \
  -d '{"video_frame": 100, "uwb_timestamp": 301564}'
```

**Use Case**: Align video timeline with UWB data timeline.

**Timestamp Calculation**:
```
For frame F:
uwb_timestamp = sync_uwb + (F - sync_frame) × (1_000_000 / video_fps)
```

---

### 3. Delete Sync Point

Remove saved sync point.

**Endpoint**: `DELETE /api/tracking/sync`

**Response**:
```json
{
  "status": "success",
  "message": "Sync point deleted"
}
```

**Example**:
```bash
curl -X DELETE http://localhost:8000/api/tracking/sync
```

---

### 4. Process Video

Run YOLO detection and ByteTrack tracking on video frames.

**Endpoint**: `POST /api/tracking/process`

**Query Parameters**:
- `start_frame` (optional): Starting frame (default: 0)
- `end_frame` (optional): Ending frame (default: last frame)
- `frame_skip` (optional): Process every Nth frame (default: 5)
- `conf_threshold` (optional): YOLO confidence threshold (default: 0.5)

**Description**: Processes video frames with YOLOv11 person detection and ByteTrack tracking.

**Response**:
```json
{
  "status": "success",
  "message": "Processed 100 frames",
  "video_info": {
    "total_frames": 50000,
    "fps": 30.0,
    "processed_frames": [0, 5, 10, 15, ..., 495]
  },
  "detection_summary": {
    "frames_processed": 100,
    "total_detections": 523,
    "avg_detections_per_frame": 5.23,
    "max_detections_in_frame": 8,
    "frames_with_detections": 98
  }
}
```

**Processing Steps**:
1. Load video file
2. Extract specified frames
3. Run YOLOv11 detection (person class only)
4. Apply ByteTrack for consistent track IDs
5. Cache results to `data/cache/detections/`
6. Return summary

**Example**:
```bash
# Process first 500 frames, every 5th frame
curl -X POST "http://localhost:8000/api/tracking/process?start_frame=0&end_frame=500&frame_skip=5"

# Process with custom confidence threshold
curl -X POST "http://localhost:8000/api/tracking/process?start_frame=0&end_frame=500&frame_skip=5&conf_threshold=0.7"
```

**Processing Time**:
- 500 frames, skip=5 → ~2-5 minutes (100 frames processed)
- Depends on hardware (GPU recommended)
- Results are cached for reuse

**Error Response** (400):
```json
{
  "detail": "Error: Video file not found"
}
```

---

### 5. Get Detection Results

Retrieve YOLO detection results for a specific frame.

**Endpoint**: `GET /api/tracking/results/{frame}`

**Parameters**:
- `frame` (path, required): Frame number

**Response**:
```json
{
  "frame": 150,
  "detections": [
    {
      "bbox": [120.5, 200.3, 180.2, 350.7],
      "confidence": 0.87,
      "class_id": 0,
      "track_id": 1,
      "bbox_bottom_center": [150.35, 350.7]
    },
    {
      "bbox": [500.1, 150.2, 560.8, 320.5],
      "confidence": 0.92,
      "class_id": 0,
      "track_id": 2,
      "bbox_bottom_center": [530.45, 320.5]
    }
  ]
}
```

**Detection Object**:
- `bbox`: Bounding box [x1, y1, x2, y2] in pixels
- `confidence`: Detection confidence (0-1)
- `class_id`: Object class (0=person)
- `track_id`: Persistent tracking ID
- `bbox_bottom_center`: Bottom center point (for matching)

**Example**:
```bash
curl http://localhost:8000/api/tracking/results/150
```

**Error Response** (404):
```json
{
  "detail": "No detections found for frame 150. Process video first."
}
```

---

### 6. Get Matched Tags

Get tag-to-player associations for a specific frame.

**Endpoint**: `GET /api/tracking/matched/{frame}`

**Parameters**:
- `frame` (path, required): Frame number

**Description**:
1. Calculates UWB timestamp for frame using sync point
2. Retrieves all tags at that timestamp
3. Loads detections for frame
4. Transforms player positions to court coordinates
5. Matches players to nearest tags within 200cm

**Response**:
```json
{
  "status": "success",
  "frame": 150,
  "timestamp": 1969897,
  "tags": [
    {
      "tag_id": 1587672,
      "x": 523.5,
      "y": 892.3,
      "timestamp": 1969900,
      "datetime": "2025-11-22 02:27:01.970",
      "matched_player": {
        "track_id": 1,
        "distance_cm": 45.2,
        "video_position": [150.35, 350.7],
        "court_position": [520.1, 890.5]
      }
    },
    {
      "tag_id": 1587036,
      "x": 1234.0,
      "y": 456.8,
      "timestamp": 1969905,
      "datetime": "2025-11-22 02:27:01.970",
      "matched_player": null
    }
  ],
  "unmatched_players": [
    {
      "track_id": 3,
      "video_position": [800.0, 400.0],
      "court_position": [1500.0, 800.0],
      "reason": "No tag within 200cm threshold"
    }
  ]
}
```

**Matching Algorithm**:
1. For each detected player:
   - Get bottom center of bounding box (video coords)
   - Transform to court coordinates using homography
   - Calculate distance to all tags
   - Match to closest tag if distance < 200cm
2. Tag can match to only one player (closest)

**Example**:
```bash
curl http://localhost:8000/api/tracking/matched/150
```

**Error Response** (400):
```json
{
  "detail": "Error: 400: Calibration required"
}
```

**Error Response** (400):
```json
{
  "detail": "Error: 400: Sync point required"
}
```

---

## Error Handling

### Error Response Format

All errors return JSON with `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

#### 400 Bad Request

**Missing Calibration**:
```json
{
  "detail": "Error: 400: Calibration required"
}
```

**Missing Sync Point**:
```json
{
  "detail": "Error: 400: Sync point required"
}
```

**Invalid Parameters**:
```json
{
  "detail": "At least 10 point pairs required for calibration"
}
```

#### 404 Not Found

**Tag Not Found**:
```json
{
  "detail": "Tag 9999999 not found"
}
```

**Frame Not Processed**:
```json
{
  "detail": "No detections found for frame 150. Process video first."
}
```

#### 500 Internal Server Error

**File System Error**:
```json
{
  "detail": "Error: Failed to read video file"
}
```

**Processing Error**:
```json
{
  "detail": "Error: YOLO model failed to load"
}
```

---

## Data Models

### TagPosition

```typescript
{
  timestamp: number;      // UWB timestamp (microseconds)
  x: number;             // X coordinate (cm)
  y: number;             // Y coordinate (cm)
  datetime: string;      // ISO 8601 datetime
}
```

### TagData

```typescript
{
  tag_id: number;
  positions: TagPosition[];
}
```

### CourtBounds

```typescript
{
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
  width: number;
  height: number;
}
```

### CourtGeometry

```typescript
{
  type: "CourtGeometry";
  bounds: CourtBounds;
  polylines: number[][][];    // Array of polylines (array of points)
  lines: number[][][];        // Array of lines (2 points each)
  circles: {
    center: number[];         // [x, y]
    radius: number;
  }[];
}
```

### Detection

```typescript
{
  bbox: number[];              // [x1, y1, x2, y2]
  confidence: number;          // 0-1
  class_id: number;            // 0 for person
  track_id?: number;           // Optional tracking ID
  bbox_bottom_center: number[]; // [x, y]
}
```

### Track

```typescript
{
  track_id: number;
  bbox: number[];
  confidence: number;
  class_id: number;
  bbox_bottom_center: number[];
}
```

### Calibration

```typescript
{
  homography: number[][];      // 3x3 matrix
  court_points: number[][];    // Array of [x, y]
  video_points: number[][];    // Array of [x, y]
  timestamp: string;           // ISO 8601
}
```

### SyncPoint

```typescript
{
  video_frame: number;         // Frame number
  uwb_timestamp: number;       // UWB timestamp
}
```

### TagMatch

```typescript
{
  tag_id: number;
  x: number;                   // Court X (cm)
  y: number;                   // Court Y (cm)
  timestamp: number;
  datetime: string;
  matched_player: {
    track_id: number;
    distance_cm: number;
    video_position: number[];  // [x, y]
    court_position: number[];  // [x, y]
  } | null;
}
```

---

## API Usage Examples

### Complete Workflow via API

```bash
# 1. Process UWB log data
curl -X POST http://localhost:8000/api/tags/process

# 2. Get list of tags
curl http://localhost:8000/api/tags/list

# 3. Get specific tag data
curl http://localhost:8000/api/tags/1587672

# 4. Get court geometry
curl http://localhost:8000/api/court/geometry

# 5. Get court image
curl http://localhost:8000/api/court/image -o court.png

# 6. Submit calibration points
curl -X POST http://localhost:8000/api/calibration/points \
  -H "Content-Type: application/json" \
  -d @calibration_points.json

# 7. Set sync point
curl -X POST http://localhost:8000/api/tracking/sync \
  -H "Content-Type: application/json" \
  -d '{"video_frame": 100, "uwb_timestamp": 301564}'

# 8. Process video
curl -X POST "http://localhost:8000/api/tracking/process?start_frame=0&end_frame=500&frame_skip=5"

# 9. Get matched tags for a frame
curl http://localhost:8000/api/tracking/matched/150
```

### Python Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Process tags
response = requests.post(f"{BASE_URL}/api/tags/process")
print(response.json())

# Get all tags
tags = requests.get(f"{BASE_URL}/api/tags/list").json()
print(f"Found {len(tags)} tags")

# Get specific tag
tag_data = requests.get(f"{BASE_URL}/api/tags/{tags[0]}").json()
print(f"Tag {tag_data['tag_id']} has {len(tag_data['positions'])} positions")

# Set sync point
sync_data = {
    "video_frame": 100,
    "uwb_timestamp": 301564
}
response = requests.post(f"{BASE_URL}/api/tracking/sync", json=sync_data)
print(response.json())

# Get matched tags for frame
matches = requests.get(f"{BASE_URL}/api/tracking/matched/150").json()
print(f"Frame 150: {len(matches['tags'])} tags matched")
```

### JavaScript Example

```javascript
const BASE_URL = 'http://localhost:8000';

// Process tags
async function processTags() {
  const response = await fetch(`${BASE_URL}/api/tags/process`, {
    method: 'POST'
  });
  const data = await response.json();
  console.log(`Processed ${data.tags_processed} tags`);
}

// Get tag data
async function getTagData(tagId) {
  const response = await fetch(`${BASE_URL}/api/tags/${tagId}`);
  const data = await response.json();
  return data;
}

// Submit calibration
async function submitCalibration(courtPoints, videoPoints) {
  const response = await fetch(`${BASE_URL}/api/calibration/points`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      court_points: courtPoints,
      video_points: videoPoints
    })
  });
  return await response.json();
}

// Get matched tags
async function getMatchedTags(frame) {
  const response = await fetch(`${BASE_URL}/api/tracking/matched/${frame}`);
  return await response.json();
}
```

---

## Testing APIs

### Using curl

```bash
# Test server is running
curl http://localhost:8000/

# Test all endpoints
curl http://localhost:8000/docs
```

### Using Python requests

```bash
pip install requests
python
>>> import requests
>>> r = requests.get('http://localhost:8000/api/tags/list')
>>> r.json()
```

### Using Postman

1. Import collection from Swagger UI
2. Set base URL: `http://localhost:8000`
3. Test each endpoint

---

For detailed usage workflows, see **USER_GUIDE.md**. For system architecture, see **SYSTEM_ARCHITECTURE.md**.
