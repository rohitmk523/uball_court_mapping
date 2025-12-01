# Future Improvements

Comprehensive list of potential enhancements, features, and optimizations for the Basketball Court Tracking System.

## Table of Contents

1. [High Priority Improvements](#high-priority-improvements)
2. [User Experience Enhancements](#user-experience-enhancements)
3. [Performance Optimizations](#performance-optimizations)
4. [Advanced Features](#advanced-features)
5. [Data Analysis & Visualization](#data-analysis--visualization)
6. [System Architecture](#system-architecture)
7. [Integration & Export](#integration--export)
8. [Research & Experimental](#research--experimental)

---

## High Priority Improvements

### 1. Video Frame Display in Tracking Page

**Current State**: Only court visualization is shown
**Improvement**: Display video frame with bounding boxes

**Implementation**:
```javascript
// Add video canvas to tracking.html
<canvas id="videoCanvas" width="1920" height="1080"></canvas>

// In tracking.js
async function displayFrame(frameNumber) {
    // Fetch frame image
    const response = await fetch(`/api/video/frame/${frameNumber}`);
    const blob = await response.blob();
    const img = await createImageBitmap(blob);

    // Draw on canvas
    const ctx = videoCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Draw bounding boxes
    const detections = await fetch(`/api/tracking/results/${frameNumber}`).json();
    for (const det of detections.detections) {
        drawBoundingBox(ctx, det.bbox, det.track_id, det.matched_tag_id);
    }
}
```

**Backend**:
```python
# Add endpoint in tracking.py
@router.get("/frame/{frame_number}")
async def get_video_frame(frame_number: int):
    """Extract and return specific video frame."""
    cap = cv2.VideoCapture(str(VIDEO_FILE))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        raise HTTPException(404, "Frame not found")

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
```

**Benefits**:
- Visual verification of detections
- Better understanding of matching results
- Easier debugging of calibration issues

**Estimated Effort**: 4-6 hours

---

### 2. Real-time Progress Updates for Video Processing

**Current State**: Loading overlay with no progress indication
**Improvement**: Show frame-by-frame progress

**WebSocket Implementation**:
```python
# backend: websocket endpoint
from fastapi import WebSocket

@app.websocket("/ws/processing")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    for frame in frame_range:
        # Process frame
        detections = detect(frame)

        # Send progress
        await websocket.send_json({
            "frame": frame,
            "total": total_frames,
            "progress": frame / total_frames,
            "detections": len(detections)
        })

    await websocket.close()
```

```javascript
// frontend: connect to websocket
const ws = new WebSocket('ws://localhost:8000/ws/processing');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateProgressBar(data.progress * 100);
    updateStatus(`Frame ${data.frame}/${data.total} - ${data.detections} detections`);
};
```

**UI Enhancement**:
```html
<div class="progress-bar-container">
    <div class="progress-bar" id="processingProgress"></div>
</div>
<p>Frame: <span id="currentFrame">0</span> / <span id="totalFrames">0</span></p>
<p>Detections: <span id="detectionCount">0</span></p>
<p>Estimated time remaining: <span id="timeRemaining">--</span></p>
```

**Benefits**:
- User knows processing is working
- Can estimate completion time
- Early detection of issues

**Estimated Effort**: 6-8 hours

---

### 3. Automatic Calibration (Feature Detection)

**Current State**: Manual point selection (10-15 points)
**Improvement**: Automatic detection of court lines

**Algorithm**:
```python
import cv2

def auto_calibrate(video_frame, court_image):
    # 1. Detect lines in video using Hough transform
    gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    # 2. Detect lines in court image
    court_gray = cv2.cvtColor(court_image, cv2.COLOR_BGR2GRAY)
    court_edges = cv2.Canny(court_gray, 50, 150)
    court_lines = cv2.HoughLinesP(court_edges, 1, np.pi/180, threshold=100)

    # 3. Find line intersections (potential points)
    video_points = find_intersections(lines)
    court_points = find_intersections(court_lines)

    # 4. Match corresponding intersections
    matched_points = match_intersections(video_points, court_points)

    # 5. Compute homography
    H, mask = cv2.findHomography(matched_points[0], matched_points[1],
                                  cv2.RANSAC, 5.0)

    return H
```

**Challenges**:
- Court lines may be occluded by players
- Lighting variations
- Court markings may be faded

**Hybrid Approach**:
- Start with automatic detection
- Allow manual refinement
- Suggest good points to user

**Benefits**:
- Faster calibration
- More consistent results
- Reduces user error

**Estimated Effort**: 15-20 hours

---

### 4. Temporal Consistency in Matching

**Current State**: Each frame matched independently
**Improvement**: Use track history for better matching

**Algorithm**:
```python
class TemporalMatcher:
    def __init__(self):
        self.track_to_tag_history = {}  # {track_id: [tag_id, ...]}

    def match_with_history(self, tracks, tags):
        matches = []

        for track in tracks:
            # Get previous tag for this track
            prev_tags = self.track_to_tag_history.get(track.track_id, [])

            # Find closest tag
            closest_tag, distance = find_closest(track, tags)

            # If previous tag is close enough, prefer it
            if prev_tags and prev_tags[-1] in tags:
                prev_tag = get_tag(prev_tags[-1])
                prev_distance = calculate_distance(track, prev_tag)

                if prev_distance < distance + 50:  # 50cm hysteresis
                    closest_tag = prev_tag
                    distance = prev_distance

            # Record match
            if distance < THRESHOLD:
                matches.append((track.track_id, closest_tag.tag_id, distance))
                self.track_to_tag_history.setdefault(
                    track.track_id, []
                ).append(closest_tag.tag_id)

        return matches
```

**Benefits**:
- Reduces tag switching (flickering)
- More stable associations
- Better handles close players

**Estimated Effort**: 4-6 hours

---

## User Experience Enhancements

### 5. Calibration Point Validation

**Feature**: Real-time feedback on point selection quality

**Indicators**:
1. **Point distribution check**:
   ```javascript
   function checkDistribution(points) {
       // Divide court into 4 quadrants
       const quadrants = [0, 0, 0, 0];
       for (const p of points) {
           const q = getQuadrant(p);
           quadrants[q]++;
       }
       // Warn if any quadrant empty
       return quadrants.every(count => count >= 2);
   }
   ```

2. **Reprojection error preview**:
   ```javascript
   async function previewCalibration(courtPts, videoPts) {
       // Send to backend
       const H = await computeHomography(courtPts, videoPts);

       // Show reprojection errors
       for (let i = 0; i < courtPts.length; i++) {
           const projected = transformPoint(courtPts[i], H);
           const error = distance(projected, videoPts[i]);

           // Color code points by error
           if (error > 10) {
               markPointAsError(i);  // Red
           } else if (error > 5) {
               markPointAsWarning(i);  // Yellow
           } else {
               markPointAsGood(i);  // Green
           }
       }
   }
   ```

3. **Point density heatmap**:
   ```javascript
   // Show which areas need more points
   function showDensityHeatmap(points, courtBounds) {
       const heatmap = createHeatmap(courtBounds);
       for (const p of points) {
           heatmap.addPoint(p, radius=200);
       }
       overlayHeatmap(heatmap);
   }
   ```

**Estimated Effort**: 6-8 hours

---

### 6. Guided Calibration Tutorial

**Feature**: Step-by-step interactive tutorial

**Steps**:
1. **Introduction**: Explain homography concept
2. **Point selection demo**: Animated example
3. **Practice mode**: User tries with hints
4. **Verification**: Check their calibration quality

**Implementation**:
```javascript
class CalibrationTutorial {
    steps = [
        {
            title: "Select Corner Points",
            description: "Start with the four corners of the court",
            hints: [
                {position: [0, 0], label: "Top-left corner"},
                {position: [2460, 0], label: "Top-right corner"},
                // ...
            ]
        },
        {
            title: "Add Center Points",
            description: "Add points near the center for better accuracy",
            // ...
        }
    ];

    showStep(stepIndex) {
        // Display instructions
        // Highlight suggested points
        // Wait for user input
        // Validate and proceed
    }
}
```

**Benefits**:
- Reduces calibration errors
- Faster onboarding
- Better user confidence

**Estimated Effort**: 8-10 hours

---

### 7. Keyboard Shortcuts

**Feature**: Quick actions via keyboard

**Shortcuts**:
```javascript
document.addEventListener('keydown', (e) => {
    switch(e.key) {
        case ' ':  // Space
            togglePlayPause();
            break;
        case 'ArrowLeft':
            previousFrame();
            break;
        case 'ArrowRight':
            nextFrame();
            break;
        case 'r':
            reset();
            break;
        case '[':
            decreaseFPS();
            break;
        case ']':
            increaseFPS();
            break;
        case 'u':  // Undo (calibration)
            undoLastPoint();
            break;
        case 'c':  // Clear (calibration)
            clearAllPoints();
            break;
    }
});
```

**UI Hint**:
```html
<div class="keyboard-hints">
    <p>Keyboard Shortcuts:</p>
    <ul>
        <li><kbd>Space</kbd> Play/Pause</li>
        <li><kbd>←</kbd> <kbd>→</kbd> Previous/Next Frame</li>
        <li><kbd>[</kbd> <kbd>]</kbd> Decrease/Increase FPS</li>
        <li><kbd>R</kbd> Reset</li>
    </ul>
</div>
```

**Estimated Effort**: 2-3 hours

---

### 8. Dark Mode

**Feature**: Dark theme for reduced eye strain

**Implementation**:
```css
/* Add to style.css */
:root {
    --bg-primary: #f5f5f5;
    --bg-secondary: white;
    --text-primary: #333;
    --text-secondary: #666;
}

[data-theme="dark"] {
    --bg-primary: #1a1a1a;
    --bg-secondary: #2d2d2d;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0a0;
}

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}
```

```javascript
// Toggle theme
function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
}

// Persist preference
const saved = localStorage.getItem('theme');
if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
}
```

**Estimated Effort**: 4-6 hours

---

## Performance Optimizations

### 9. Frame Caching & Pre-loading

**Feature**: Cache video frames for instant display

**Implementation**:
```javascript
class FrameCache {
    constructor(capacity = 100) {
        this.cache = new Map();
        this.capacity = capacity;
    }

    async get(frameNumber) {
        if (this.cache.has(frameNumber)) {
            return this.cache.get(frameNumber);
        }

        // Fetch frame
        const frame = await fetchFrame(frameNumber);

        // Add to cache
        this.cache.set(frameNumber, frame);

        // Evict old frames if needed
        if (this.cache.size > this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        return frame;
    }

    preload(frameNumbers) {
        // Preload frames in background
        for (const num of frameNumbers) {
            this.get(num);  // Async, fire and forget
        }
    }
}

// Usage
const cache = new FrameCache(100);

// Preload next 10 frames
function preloadNextFrames(currentFrame) {
    const next = Array.from({length: 10}, (_, i) => currentFrame + i + 1);
    cache.preload(next);
}
```

**Benefits**:
- Smooth playback (no loading delays)
- Better user experience
- Efficient memory usage

**Estimated Effort**: 4-6 hours

---

### 10. Server-Side Rendering for Large Datasets

**Feature**: Render tag animations on server, stream to client

**For**: Very large datasets (> 1M positions per tag)

**Implementation**:
```python
# backend: render animation frames server-side
from PIL import Image, ImageDraw

@router.get("/animation/frame/{timestamp}")
async def render_animation_frame(timestamp: int):
    # Get court image
    img = Image.open(COURT_IMAGE_FILE)
    draw = ImageDraw.Draw(img)

    # Get all tags at timestamp
    tags = await get_tags_at_timestamp(timestamp)

    # Draw tags
    for tag in tags:
        x, y = court_to_image_coords(tag.x, tag.y)
        color = get_tag_color(tag.tag_id)
        draw.circle((x, y), radius=10, fill=color)

    # Return as JPEG
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return Response(content=buffer.getvalue(), media_type="image/jpeg")
```

**Benefits**:
- Handles unlimited data size
- Reduces client memory usage
- Faster initial load

**Trade-offs**:
- Higher server load
- Network bandwidth for image streaming

**Estimated Effort**: 8-10 hours

---

### 11. Database Backend

**Feature**: Replace file storage with database

**Schema**:
```sql
CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY
);

CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    tag_id INTEGER REFERENCES tags(tag_id),
    timestamp BIGINT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    datetime TIMESTAMP NOT NULL,
    INDEX idx_tag_timestamp (tag_id, timestamp)
);

CREATE TABLE calibrations (
    id SERIAL PRIMARY KEY,
    homography JSONB NOT NULL,
    court_points JSONB NOT NULL,
    video_points JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE detections (
    frame_number INTEGER PRIMARY KEY,
    detections JSONB NOT NULL
);
```

**Benefits**:
- Faster queries (indexed)
- Concurrent access
- Transactional updates
- Better scalability

**Migration Script**:
```python
import json
import psycopg2

def migrate_tags_to_db():
    conn = psycopg2.connect("dbname=court_tracking")
    cur = conn.cursor()

    for tag_file in Path("data/output/tags").glob("*.json"):
        tag_data = json.load(open(tag_file))

        # Insert tag
        cur.execute("INSERT INTO tags (tag_id) VALUES (%s)",
                   (tag_data['tag_id'],))

        # Insert positions
        for pos in tag_data['positions']:
            cur.execute("""
                INSERT INTO positions (tag_id, timestamp, x, y, datetime)
                VALUES (%s, %s, %s, %s, %s)
            """, (tag_data['tag_id'], pos['timestamp'],
                  pos['x'], pos['y'], pos['datetime']))

    conn.commit()
```

**Estimated Effort**: 15-20 hours

---

## Advanced Features

### 12. Multiple Camera Support

**Feature**: Combine data from multiple camera views

**Architecture**:
```
Camera 1 (Hoop) ─┐
Camera 2 (Side)  ├─→ Fused Tracking
Camera 3 (End)   ─┘
```

**Implementation**:
```python
class MultiCameraTracker:
    def __init__(self):
        self.cameras = []
        self.calibrations = []  # One per camera

    def add_camera(self, video_file, calibration):
        self.cameras.append({
            'video': video_file,
            'calibration': calibration
        })

    def fuse_detections(self, frame_number):
        all_detections = []

        # Get detections from each camera
        for cam in self.cameras:
            dets = detect_in_video(cam['video'], frame_number)

            # Transform to court coordinates
            court_dets = [
                transform_detection(d, cam['calibration'])
                for d in dets
            ]

            all_detections.extend(court_dets)

        # Cluster detections (same player seen by multiple cameras)
        fused = cluster_detections(all_detections, threshold=100)

        return fused
```

**Clustering**:
```python
def cluster_detections(detections, threshold):
    """Group detections that are close in court coords."""
    clusters = []

    for det in detections:
        # Find closest cluster
        closest = None
        min_dist = float('inf')

        for cluster in clusters:
            dist = distance(det.position, cluster.center)
            if dist < min_dist:
                min_dist = dist
                closest = cluster

        # Add to cluster or create new
        if min_dist < threshold:
            closest.add(det)
        else:
            clusters.append(Cluster([det]))

    return clusters
```

**Benefits**:
- Better coverage (no blind spots)
- Handles occlusions
- More robust tracking

**Estimated Effort**: 25-30 hours

---

### 13. Player Re-identification

**Feature**: Recognize players across occlusions and camera views

**Approach**: Use appearance features (jersey color, number, etc.)

**Implementation**:
```python
from transformers import CLIPModel, CLIPProcessor

class PlayerReID:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embeddings = {}  # {track_id: embedding}

    def extract_embedding(self, frame, bbox):
        """Extract appearance embedding for player crop."""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        # Preprocess
        inputs = self.processor(images=crop, return_tensors="pt")

        # Get embedding
        outputs = self.model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy()

        return embedding

    def match_player(self, embedding):
        """Find most similar existing player."""
        best_match = None
        best_similarity = 0

        for track_id, stored_emb in self.embeddings.items():
            similarity = cosine_similarity(embedding, stored_emb)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = track_id

        if best_similarity > 0.8:  # Threshold
            return best_match
        else:
            return None  # New player
```

**Benefits**:
- Maintain track IDs through occlusions
- Better multi-camera association
- Handle player substitutions

**Estimated Effort**: 20-25 hours

---

### 14. Play Classification

**Feature**: Automatically classify basketball plays

**Categories**:
- Offensive plays (fast break, half-court set)
- Defensive plays (man-to-man, zone)
- Transitions
- Set pieces

**Implementation**:
```python
class PlayClassifier:
    def __init__(self):
        self.model = load_trained_model('play_classifier.pth')

    def classify_sequence(self, positions, duration=5):
        """
        Classify play from position sequence.

        positions: List of player positions over time
        duration: Seconds to consider
        """
        # Extract features
        features = extract_features(positions, duration)

        # Features:
        # - Team formation (spread, concentrated)
        # - Movement patterns (speeds, directions)
        # - Court zones occupied
        # - Possession changes

        # Classify
        probabilities = self.model(features)
        play_type = np.argmax(probabilities)

        return {
            'type': PLAY_TYPES[play_type],
            'confidence': probabilities[play_type],
            'start_time': positions[0].timestamp,
            'end_time': positions[-1].timestamp
        }
```

**Training Data**: Manually labeled sequences

**Benefits**:
- Automatic play analysis
- Search by play type
- Statistical insights

**Estimated Effort**: 40-50 hours (including data collection)

---

### 15. Heat Maps & Trajectory Visualization

**Feature**: Visualize player movement patterns

**Types**:
1. **Density Heatmap**: Where players spend most time
2. **Speed Heatmap**: Movement speed in different zones
3. **Trajectories**: Paths taken by players

**Implementation**:
```python
import matplotlib.pyplot as plt
import numpy as np

def generate_heatmap(positions, court_bounds):
    """Generate density heatmap from positions."""
    # Create 2D histogram
    x_coords = [p.x for p in positions]
    y_coords = [p.y for p in positions]

    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=[50, 35],  # Resolution
        range=[[0, court_bounds.max_x], [0, court_bounds.max_y]]
    )

    return heatmap

def plot_heatmap(heatmap, court_image):
    """Overlay heatmap on court image."""
    plt.imshow(court_image, alpha=0.5)
    plt.imshow(heatmap.T, origin='lower',
               cmap='hot', alpha=0.6, interpolation='gaussian')
    plt.colorbar(label='Time Spent (seconds)')
    plt.title('Player Movement Heatmap')
    plt.show()
```

**Interactive Visualization**:
```javascript
// Canvas overlay with slider
class HeatmapVisualizer {
    render(timeRange) {
        // Get positions in time range
        const positions = getPositionsInRange(timeRange);

        // Generate heatmap
        const heatmap = computeHeatmap(positions);

        // Render on canvas
        renderHeatmapOverlay(this.canvas, heatmap);
    }
}

// Allow time range selection
<input type="range" id="startTime" min="0" max="duration">
<input type="range" id="endTime" min="0" max="duration">
```

**Benefits**:
- Understand movement patterns
- Identify preferred zones
- Analyze offensive/defensive positioning

**Estimated Effort**: 10-15 hours

---

## Data Analysis & Visualization

### 16. Statistics Dashboard

**Feature**: Comprehensive statistics and analytics

**Metrics**:

**Player Statistics**:
```python
- Total distance traveled
- Average speed
- Maximum speed
- Time in each court zone
- Possession time
- Shot attempts (if integrated with ball tracking)
```

**Team Statistics**:
```python
- Formation metrics (spread, compactness)
- Ball possession time
- Transition speed
- Defensive pressure
```

**Dashboard UI**:
```html
<div class="stats-dashboard">
    <div class="stat-card">
        <h3>Player 1 (Tag 1587672)</h3>
        <p>Distance: <strong>1,245 m</strong></p>
        <p>Avg Speed: <strong>3.2 m/s</strong></p>
        <p>Max Speed: <strong>6.8 m/s</strong></p>
    </div>

    <div class="stat-card">
        <h3>Team Performance</h3>
        <p>Possession: <strong>58%</strong></p>
        <p>Avg Formation Spread: <strong>12.5 m</strong></p>
    </div>

    <div class="chart-container">
        <canvas id="speedChart"></canvas>
    </div>
</div>
```

**Charts** (using Chart.js):
```javascript
// Speed over time
new Chart(ctx, {
    type: 'line',
    data: {
        labels: timestamps,
        datasets: [{
            label: 'Speed (m/s)',
            data: speeds,
            borderColor: 'blue'
        }]
    }
});
```

**Estimated Effort**: 20-25 hours

---

### 17. Export to Common Formats

**Feature**: Export data for external analysis

**Formats**:
1. **CSV**: For spreadsheet analysis
2. **JSON**: For programmatic use
3. **Video**: Annotated video with overlays
4. **PDF Report**: Automated report generation

**CSV Export**:
```python
@router.get("/export/csv")
async def export_csv(start_time: int, end_time: int):
    """Export position data as CSV."""
    positions = get_positions_in_range(start_time, end_time)

    csv_data = "timestamp,tag_id,x,y,matched_player\n"
    for pos in positions:
        csv_data += f"{pos.timestamp},{pos.tag_id},{pos.x},{pos.y},{pos.matched_player}\n"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=positions_{start_time}_{end_time}.csv"
        }
    )
```

**Video Export**:
```python
def export_annotated_video(video_path, output_path, start_frame, end_frame):
    """Export video with bounding boxes and tags overlaid."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1920, 1080))

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw detections
        detections = get_detections(frame_num)
        for det in detections:
            draw_bbox(frame, det.bbox, det.track_id)
            if det.matched_tag_id:
                draw_tag_label(frame, det.bbox, det.matched_tag_id)

        out.write(frame)

    cap.release()
    out.release()
```

**PDF Report**:
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_report(output_path, stats):
    """Generate PDF report with statistics and visualizations."""
    c = canvas.Canvas(output_path, pagesize=letter)

    # Title
    c.drawString(100, 750, "Basketball Tracking Report")

    # Statistics
    c.drawString(100, 700, f"Date: {stats['date']}")
    c.drawString(100, 680, f"Duration: {stats['duration']} minutes")

    # Include charts (as images)
    c.drawImage(stats['heatmap_image'], 100, 400, width=400, height=250)

    c.save()
```

**Estimated Effort**: 15-20 hours

---

## System Architecture

### 18. Microservices Architecture

**Feature**: Split into independent services

**Services**:
```
┌─────────────────┐
│  API Gateway    │ (FastAPI)
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼──┐ ┌───▼───┐ ┌──▼──┐
│ Tags  │ │Court│ │Tracking│ │Calib│
│Service│ │Serv.│ │Service │ │Serv.│
└───────┘ └─────┘ └────────┘ └─────┘
```

**Benefits**:
- Independent scaling
- Technology flexibility
- Better fault isolation
- Easier deployment

**Implementation**:
```python
# tags_service.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/tags/{tag_id}")
async def get_tag(tag_id: int):
    # ...

# Run on port 8001
uvicorn.run(app, port=8001)

# API Gateway routes requests
@gateway.get("/api/tags/{tag_id}")
async def proxy_to_tags(tag_id: int):
    response = httpx.get(f"http://tags-service:8001/tags/{tag_id}")
    return response.json()
```

**Estimated Effort**: 30-40 hours

---

### 19. Containerization (Docker)

**Feature**: Package as Docker containers

**Dockerfile**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - VIDEO_FILE=/app/data/video.mp4

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: court_tracking
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

**Benefits**:
- Consistent environment
- Easy deployment
- Portable across systems
- Simplified scaling

**Estimated Effort**: 8-10 hours

---

## Integration & Export

### 20. REST API Client Library

**Feature**: Python/JavaScript client for API

**Python Client**:
```python
# court_tracking_client.py
import requests

class CourtTrackingClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def process_tags(self):
        """Process UWB log file."""
        response = requests.post(f"{self.base_url}/api/tags/process")
        return response.json()

    def get_tag(self, tag_id):
        """Get tag position history."""
        response = requests.get(f"{self.base_url}/api/tags/{tag_id}")
        return response.json()

    def calibrate(self, court_points, video_points):
        """Submit calibration points."""
        response = requests.post(
            f"{self.base_url}/api/calibration/points",
            json={
                "court_points": court_points,
                "video_points": video_points
            }
        )
        return response.json()

    def process_video(self, start=0, end=500, skip=5):
        """Process video with YOLO."""
        response = requests.post(
            f"{self.base_url}/api/tracking/process",
            params={"start_frame": start, "end_frame": end, "frame_skip": skip}
        )
        return response.json()

# Usage
client = CourtTrackingClient()
client.process_tags()
tag_data = client.get_tag(1587672)
```

**JavaScript Client**:
```javascript
// CourtTrackingClient.js
class CourtTrackingClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async processTags() {
        const response = await fetch(`${this.baseUrl}/api/tags/process`, {
            method: 'POST'
        });
        return await response.json();
    }

    async getTag(tagId) {
        const response = await fetch(`${this.baseUrl}/api/tags/${tagId}`);
        return await response.json();
    }

    // ... more methods
}

// Usage
const client = new CourtTrackingClient();
await client.processTags();
const tagData = await client.getTag(1587672);
```

**Estimated Effort**: 6-8 hours

---

## Research & Experimental

### 21. Ball Tracking

**Feature**: Track basketball position

**Challenges**:
- Small object (difficult to detect)
- Fast movement (motion blur)
- Occlusions by players

**Approach**:
```python
# Fine-tune YOLO on basketball dataset
from ultralytics import YOLO

# Collect basketball images
# Annotate ball positions
# Train model
model = YOLO('yolo11n.pt')
model.train(data='basketball.yaml', epochs=100)

# Use trained model
ball_model = YOLO('basketball_ball.pt')
results = ball_model(frame)

# Track ball trajectory
trajectory = []
for frame in frames:
    balls = ball_model(frame)
    if len(balls) > 0:
        trajectory.append(balls[0].position)
```

**Benefits**:
- Complete game analysis
- Possession tracking
- Shot detection

**Estimated Effort**: 50-60 hours (including data collection)

---

### 22. Pose Estimation

**Feature**: Detect player body pose

**Use Cases**:
- Shooting form analysis
- Injury risk assessment
- Movement quality

**Implementation**:
```python
from ultralytics import YOLO

# Use pose estimation model
pose_model = YOLO('yolo11n-pose.pt')

# Detect keypoints
results = pose_model(frame)

# Keypoints: [nose, eyes, ears, shoulders, elbows, wrists,
#             hips, knees, ankles]
for person in results:
    keypoints = person.keypoints
    # Analyze pose
    shooting_form = analyze_shooting_form(keypoints)
```

**Estimated Effort**: 30-40 hours

---

### 23. Event Detection

**Feature**: Automatically detect game events

**Events**:
- Shots
- Passes
- Rebounds
- Turnovers
- Fouls

**Approach**:
```python
class EventDetector:
    def detect_shot(self, ball_trajectory, player_poses):
        """Detect shot attempt from ball trajectory and player pose."""
        # Check for:
        # 1. Ball moving upward
        # 2. Player in shooting pose
        # 3. Ball trajectory toward basket

        if (ball_trajectory.is_ascending() and
            player_poses.has_shooting_form() and
            ball_trajectory.toward_basket()):
            return Event(type='shot', confidence=0.9)

    def detect_pass(self, ball_trajectory, player_positions):
        """Detect pass from ball movement between players."""
        # Check for:
        # 1. Ball moving from player A to player B
        # 2. Change in possession

        if ball_trajectory.connects_players():
            return Event(type='pass', confidence=0.85)
```

**Benefits**:
- Automatic game logging
- Statistical analysis
- Highlight generation

**Estimated Effort**: 40-50 hours

---

## Implementation Priority

### Phase 1 (High Priority - Next 3 months)
1. ✅ Video frame display in tracking page
2. ✅ Real-time progress updates
3. ✅ Temporal consistency in matching
4. ✅ Calibration point validation
5. ✅ Keyboard shortcuts

### Phase 2 (Medium Priority - 3-6 months)
6. ⬜ Automatic calibration
7. ⬜ Frame caching & pre-loading
8. ⬜ Dark mode
9. ⬜ Heat maps & trajectories
10. ⬜ Statistics dashboard

### Phase 3 (Advanced Features - 6-12 months)
11. ⬜ Database backend
12. ⬜ Multiple camera support
13. ⬜ Player re-identification
14. ⬜ Export to common formats
15. ⬜ Guided calibration tutorial

### Phase 4 (Research - 12+ months)
16. ⬜ Ball tracking
17. ⬜ Pose estimation
18. ⬜ Event detection
19. ⬜ Play classification
20. ⬜ Microservices architecture

---

## Conclusion

This document outlines a comprehensive roadmap for enhancing the Basketball Court Tracking System. Priorities should be adjusted based on:
- User feedback
- Resource availability
- Research opportunities
- Business requirements

For current system capabilities, see:
- **USER_GUIDE.md** - How to use existing features
- **API_REFERENCE.md** - Current API documentation
- **TECHNICAL_DETAILS.md** - Implementation details
