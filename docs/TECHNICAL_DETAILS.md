# Technical Details

In-depth technical documentation covering algorithms, coordinate systems, and implementation details.

## Table of Contents

1. [Coordinate Systems](#coordinate-systems)
2. [Homography Transformation](#homography-transformation)
3. [YOLOv11 Detection](#yolov11-detection)
4. [ByteTrack Tracking](#bytetrack-tracking)
5. [Tag-to-Player Matching](#tag-to-player-matching)
6. [Timestamp Synchronization](#timestamp-synchronization)
7. [Performance Optimization](#performance-optimization)
8. [Data Formats](#data-formats)
9. [Mathematical Foundations](#mathematical-foundations)

---

## Coordinate Systems

The system uses three different coordinate systems that must be properly transformed between each other.

### 1. Court Coordinate System (UWB)

**Origin**: Bottom-left corner of the court
**Units**: Centimeters (cm)
**Axes**:
- X-axis: Points RIGHT (parallel to baseline)
- Y-axis: Points UP (parallel to sideline)

**Range**:
```
X: 0 to 2460.2 cm (0 to ~24.6 meters)
Y: 0 to 1730.8 cm (0 to ~17.3 meters)
```

**Standard Basketball Court**:
```
Full court: 28.65m × 15.24m (94 ft × 50 ft)
Our court: 24.6m × 17.3m (smaller, recreational)
```

**Example Position**:
```python
tag_position = (1230, 865)  # Center of court in cm
# X = 1230 cm = 12.3 meters from left
# Y = 865 cm = 8.65 meters from bottom
```

**Visualization**:
```
            Y (1730.8 cm)
            ↑
            |
            |    • Tag (1230, 865)
            |
(0,0) ------+----------→ X (2460.2 cm)
```

### 2. Video Coordinate System (Camera)

**Origin**: Top-left corner of video frame
**Units**: Pixels (px)
**Axes**:
- X-axis: Points RIGHT
- Y-axis: Points DOWN

**Range**: Depends on video resolution
```
Example: 1920×1080 video
X: 0 to 1920 px
Y: 0 to 1080 px
```

**Pixel Position**:
```python
video_position = (960, 540)  # Center of 1920×1080 video
# X = 960 pixels from left
# Y = 540 pixels from top
```

**Visualization**:
```
(0,0) ------→ X (1920 px)
  |
  |     • Player (960, 540)
  |
  ↓
Y (1080 px)
```

**Important Note**: Y-axis is **inverted** compared to court system!

### 3. Canvas Coordinate System (Browser)

**Origin**: Top-left corner of canvas element
**Units**: Pixels (scaled)
**Axes**:
- X-axis: Points RIGHT
- Y-axis: Points DOWN

**Transformation from Court to Canvas**:
```javascript
// Scale and flip Y-axis
canvasX = courtX × scale + offsetX
canvasY = -courtY × scale + offsetY  // Note the minus sign!
```

**Example**:
```javascript
// Court: (1230, 865) cm
// Canvas: 1000×800 px
// Scale: 2 px/cm
// Offset: (100, 700)

canvasX = 1230 × 2 + 100 = 2560 px
canvasY = -(865 × 2) + 700 = -1030 px (clipped to canvas bounds)
```

---

## Homography Transformation

Homography is a projective transformation that maps points from one plane to another.

### Mathematical Representation

A homography is represented by a 3×3 matrix **H**:

```
H = | h11  h12  h13 |
    | h21  h22  h23 |
    | h31  h32  h33 |
```

### Point Transformation

Transform point from court (xc, yc) to video (xv, yv):

**Homogeneous Coordinates**:
```
[xv]     [xc]
[yv]  =  [yc]  × H
[1 ]     [1 ]
```

**Full Equation**:
```
xv' = h11×xc + h12×yc + h13
yv' = h21×xc + h22×yc + h23
w'  = h31×xc + h32×yc + h33

xv = xv' / w'
yv = yv' / w'
```

**Normalization**: Divide by w' to get actual coordinates.

### Computing Homography (RANSAC)

**Input**:
- Court points: [(xc1, yc1), (xc2, yc2), ..., (xcn, ycn)]
- Video points: [(xv1, yv1), (xv2, yv2), ..., (xvn, yvn)]

**Algorithm**: `cv2.findHomography()`

```python
court_pts = np.array(court_points, dtype=np.float32)
video_pts = np.array(video_points, dtype=np.float32)

H, mask = cv2.findHomography(
    court_pts,
    video_pts,
    cv2.RANSAC,
    5.0  # RANSAC reprojection threshold
)
```

**RANSAC Steps**:
1. **Random Sample**: Select 4 random point pairs
2. **Compute H**: Calculate homography from these 4 pairs
3. **Test Inliers**: Check how many other points fit this H
4. **Repeat**: Try many random samples (e.g., 2000 iterations)
5. **Best Model**: Keep H with most inliers
6. **Refine**: Recompute H using all inliers

**Inlier Criteria**:
```
reprojection_error = distance(transformed_point, actual_point)
is_inlier = reprojection_error < 5.0 pixels
```

**Why RANSAC?**
- Robust to outliers (wrong point selections)
- Automatically detects and ignores bad points
- More reliable than least squares with noisy data

### Inverse Transformation

Transform from video to court:

```python
H_inv = np.linalg.inv(H)
court_point = transform_point(video_point, H_inv)
```

**Use Case**: Finding which court position corresponds to a detected player in video.

### Example Transformation

```python
# Given homography H and court point
court_point = np.array([1230.0, 865.0, 1.0])  # Homogeneous

# Transform to video
video_point_h = H @ court_point
video_point = video_point_h[:2] / video_point_h[2]

# Result: (960.5, 540.2) in video pixels
```

---

## YOLOv11 Detection

YOLO (You Only Look Once) is a real-time object detection algorithm.

### Model Architecture

**YOLOv11 Versions**:
```
yolo11n.pt - Nano:   3.2M params, fastest
yolo11s.pt - Small:  11.2M params
yolo11m.pt - Medium: 25.9M params
yolo11l.pt - Large:  43.7M params
yolo11x.pt - XLarge: 68.2M params, most accurate
```

**Trade-offs**:
- Nano: ~60 FPS, 90% accuracy
- XLarge: ~15 FPS, 95% accuracy

### Detection Process

**Input**: RGB image (H × W × 3)
**Output**: Bounding boxes with class predictions

**Pipeline**:
```
Image (1920×1080×3)
    ↓
Preprocessing (resize, normalize)
    ↓
Backbone Network (feature extraction)
    ↓
Neck (feature aggregation)
    ↓
Head (detection)
    ↓
Post-processing (NMS)
    ↓
Bounding Boxes [(x1, y1, x2, y2, conf, class), ...]
```

### Bounding Box Format

**YOLO Output**:
```python
{
    'bbox': [x1, y1, x2, y2],  # Top-left and bottom-right corners
    'confidence': 0.87,         # Detection confidence (0-1)
    'class_id': 0,              # Class ID (0 = person)
    'class_name': 'person'
}
```

**Coordinate Convention**:
```
(x1, y1) -------- +
   |              |
   |   Person     |
   |              |
   + -------- (x2, y2)
```

### Person Detection

**Filter for person class**:
```python
results = model(frame, conf=0.5, classes=[0])  # class 0 = person
```

**COCO Classes**: YOLOv11 is trained on COCO dataset
- Class 0: person
- Class 1: bicycle
- Class 2: car
- ... (80 classes total)

### Bottom Center Calculation

For matching tags, we use the **bottom center** of the bounding box (player's feet position).

```python
x_center = (x1 + x2) / 2
y_bottom = y2

bottom_center = (x_center, y_bottom)
```

**Rationale**:
- Feet position best matches UWB tag location
- Tags are worn at waist/chest, but we project to ground
- Bottom center is most stable across poses

### Confidence Threshold

**Adjusting Threshold**:
```python
# Lower threshold = more detections (more false positives)
results = model(frame, conf=0.3)

# Higher threshold = fewer detections (might miss players)
results = model(frame, conf=0.7)
```

**Recommended Values**:
- 0.3-0.4: Crowded scenes, many occlusions
- 0.5: Default, balanced
- 0.6-0.7: Clean scenes, avoid false positives

### Batch Processing

**Process multiple frames efficiently**:
```python
frames = [frame1, frame2, frame3, ...]
results = model(frames, stream=True)  # Generator for memory efficiency

for r in results:
    boxes = r.boxes
    # Process detections
```

---

## ByteTrack Tracking

ByteTrack maintains consistent track IDs for detected objects across frames.

### Tracking Problem

**Challenge**: Match detections across frames
- Same player should have same track ID
- Handle occlusions (player temporarily hidden)
- Handle re-identification (player reappears)

**Example**:
```
Frame 10: [Player1, Player2, Player3]
Frame 11: [Player1, Player2] (Player3 occluded)
Frame 12: [Player1, Player2, Player3] (Player3 reappears)

Desired track IDs:
Player1: ID=1 (consistent)
Player2: ID=2 (consistent)
Player3: ID=3 (maintained through occlusion)
```

### ByteTrack Algorithm

**Key Idea**: Use both high and low confidence detections

**Steps**:
1. **Separate Detections**:
   - High confidence: conf > 0.6
   - Low confidence: 0.1 < conf < 0.6

2. **Match High Confidence**:
   - Match to existing tracks using IoU
   - Update tracks with high-conf detections

3. **Match Low Confidence**:
   - Match remaining tracks to low-conf detections
   - Helps maintain tracks during occlusions

4. **Track Management**:
   - Create new tracks for unmatched high-conf detections
   - Keep lost tracks for 30 frames (track buffer)
   - Delete tracks that remain unmatched too long

### IoU Matching

**Intersection over Union (IoU)**:
```
IoU = Area(bbox1 ∩ bbox2) / Area(bbox1 ∪ bbox2)
```

**Calculate IoU**:
```python
def calculate_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union
```

**Matching Threshold**:
```python
MATCH_THRESH = 0.8  # IoU must be > 0.8 to match
```

### Hungarian Algorithm

**Optimal Assignment Problem**:
Given N detections and M tracks, find the best assignment.

**Cost Matrix**:
```
           Track1  Track2  Track3
Detection1   0.9    0.1    0.05
Detection2   0.15   0.85   0.1
Detection3   0.05   0.12   0.88
```

**Hungarian Algorithm** finds assignment that maximizes total IoU.

**Implementation**:
```python
from lapx import linear_assignment

# Create cost matrix (negative IoU for minimization)
costs = -iou_matrix

# Find optimal assignment
row_indices, col_indices = linear_assignment(costs)

# row_indices[i] is matched to col_indices[i]
```

### Track States

**Active Track**:
- Recently matched to detection
- Confidence in track position

**Lost Track**:
- Not matched for < 30 frames
- Kept in buffer for potential re-match
- Track ID preserved

**Removed Track**:
- Not matched for > 30 frames
- Track ID can be reused

### Simple Tracking Fallback

If ByteTrack unavailable, use simple position-based tracking:

```python
def simple_tracking(prev_tracks, current_detections):
    for detection in current_detections:
        # Find closest previous track
        min_distance = float('inf')
        best_track_id = None

        for track in prev_tracks:
            dist = euclidean_distance(
                detection.bbox_bottom_center,
                track.bbox_bottom_center
            )
            if dist < min_distance and dist < 100:  # 100 pixel threshold
                min_distance = dist
                best_track_id = track.track_id

        if best_track_id:
            detection.track_id = best_track_id
        else:
            detection.track_id = new_track_id()
```

---

## Tag-to-Player Matching

Matching algorithm associates UWB tags with detected players.

### Problem Statement

**Given**:
- Player positions in video (pixel coordinates)
- Tag positions in court (cm coordinates)
- Homography matrix H

**Find**: Best tag-to-player associations

### Algorithm

**Step 1: Transform Player to Court Coordinates**

```python
# Player position in video (pixels)
player_video_pos = (x_pixels, y_pixels)

# Transform to court coordinates (cm)
H_inv = np.linalg.inv(H)
player_court_pos = inverse_transform_point(player_video_pos, H_inv)
```

**Step 2: Calculate Distances**

```python
def euclidean_distance(pos1, pos2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx**2 + dy**2)

distances = []
for tag in tags:
    dist = euclidean_distance(player_court_pos, (tag.x, tag.y))
    distances.append((tag.tag_id, dist))
```

**Step 3: Find Closest Tag**

```python
THRESHOLD_CM = 200  # 2 meters

closest_tag = min(distances, key=lambda x: x[1])

if closest_tag[1] < THRESHOLD_CM:
    match = (player, closest_tag[0], closest_tag[1])
else:
    match = None  # No match within threshold
```

### Full Matching Algorithm

```python
def match_tags_to_players(players, tags, H, threshold=200):
    matches = []
    used_tags = set()

    # For each player
    for player in players:
        # Transform to court coordinates
        player_court = inverse_transform(player.position, H)

        # Find closest unused tag
        closest_tag = None
        closest_dist = float('inf')

        for tag in tags:
            if tag.tag_id in used_tags:
                continue

            dist = euclidean_distance(player_court, (tag.x, tag.y))

            if dist < closest_dist:
                closest_dist = dist
                closest_tag = tag

        # Match if within threshold
        if closest_tag and closest_dist < threshold:
            matches.append({
                'player_id': player.track_id,
                'tag_id': closest_tag.tag_id,
                'distance': closest_dist
            })
            used_tags.add(closest_tag.tag_id)

    return matches
```

### Matching Constraints

**One-to-One Matching**:
- Each tag matches at most one player
- Each player matches at most one tag

**Distance Threshold**:
- Default: 200cm (2 meters)
- Adjustable based on accuracy requirements

**Temporal Consistency** (future improvement):
- Track should maintain same tag over time
- Sudden tag switches indicate error

### Error Sources

**Calibration Error**:
- Inaccurate homography
- Wrong court-to-video mapping
- Solution: Re-calibrate with more points

**Timestamp Mismatch**:
- Wrong sync point
- Video and UWB not aligned
- Solution: Verify sync point accuracy

**Physical Distance**:
- Tag actually far from player
- Player not wearing tag
- Tag malfunction

---

## Timestamp Synchronization

Aligning video frames with UWB timestamps.

### Problem

**Two timelines**:
- Video: frames (0, 1, 2, ...) at constant FPS
- UWB: timestamps (microseconds) at variable rate

**Need**: Map frame number → UWB timestamp

### Sync Point Method

**User provides one correspondence**:
```
Sync point: frame 100 ↔ timestamp 301564
```

**Calculate timestamp for any frame**:
```python
def frame_to_timestamp(frame, sync_frame, sync_timestamp, fps):
    # Time elapsed since sync frame (in seconds)
    time_delta = (frame - sync_frame) / fps

    # Convert to microseconds
    time_delta_us = time_delta * 1_000_000

    # Add to sync timestamp
    timestamp = sync_timestamp + int(time_delta_us)

    return timestamp
```

**Example**:
```python
sync_frame = 100
sync_timestamp = 301564
fps = 30.0

# Calculate timestamp for frame 150
frame_to_timestamp(150, 100, 301564, 30.0)
# = 301564 + (150-100)/30 * 1,000,000
# = 301564 + 1,666,667
# = 1,968,231
```

### Video FPS Extraction

```python
import cv2

cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
# Returns: 30.0 (frames per second)
```

### Tag Temporal Query

**Get tags at specific timestamp**:

```python
def get_tags_at_timestamp(timestamp, tolerance=1_000_000):
    """
    Get all tags within tolerance window.

    tolerance: microseconds (default: 1 second)
    """
    min_ts = timestamp - tolerance
    max_ts = timestamp + tolerance

    matching_tags = []
    for tag in all_tags:
        for pos in tag.positions:
            if min_ts <= pos.timestamp <= max_ts:
                matching_tags.append({
                    'tag_id': tag.tag_id,
                    'x': pos.x,
                    'y': pos.y,
                    'timestamp': pos.timestamp
                })
                break  # Use first matching position

    return matching_tags
```

**Tolerance Trade-offs**:
- Small (100ms): More accurate timing, might miss tags
- Large (2s): More tags found, less accurate timing
- Default (1s): Good balance

---

## Performance Optimization

### 1. Detection Caching

**Problem**: YOLO processing is slow (minutes for 100 frames)

**Solution**: Cache results to disk

```python
cache_file = f"detections_{start}_{end}_{skip}.json"

if os.path.exists(cache_file):
    # Load from cache
    with open(cache_file, 'r') as f:
        detections = json.load(f)
else:
    # Process video
    detections = process_video()

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(detections, f)
```

**Cache Key**: Includes frame range and skip value
- Changing any parameter invalidates cache
- Delete cache files to force reprocessing

### 2. Lazy Loading

**Don't load all data at once**:

```python
# Bad: Load all tag data upfront
all_tags = [load_tag(id) for id in tag_ids]

# Good: Load on demand
def get_tag(tag_id):
    if tag_id not in cache:
        cache[tag_id] = load_tag(tag_id)
    return cache[tag_id]
```

### 3. Frame Skipping

**Process every Nth frame**:
```python
for frame_num in range(start, end, skip):
    frame = cap.read()[1]
    process_frame(frame)
```

**Trade-off**:
- Skip=1: All frames, slow, best quality
- Skip=5: 5× faster, good quality
- Skip=30: 30× faster, ~1 frame/second

### 4. GPU Acceleration

**Use CUDA for YOLO**:
```python
import torch

# Check GPU available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model on GPU
model = YOLO('yolo11n.pt')
model.to(device)

# ~5-10× speedup with GPU
```

### 5. Batch Processing

**Process multiple frames at once**:
```python
batch_size = 8
frames_batch = []

for i in range(0, len(frames), batch_size):
    batch = frames[i:i+batch_size]
    results = model(batch)  # Process batch together
```

### 6. Model Selection

**Choose appropriate model**:
```
yolo11n.pt: 3.2M params, 1.5ms/frame (GPU)
yolo11x.pt: 68M params, 8.5ms/frame (GPU)
```

**For real-time**: Use nano or small
**For accuracy**: Use medium or large

---

## Data Formats

### UWB Log Format

```
2025-11-22 02:26:40.578 | Tag 1587672 | X=454 | Y=469 | Timestamp=268737020
```

**Fields**:
- Datetime: ISO 8601 format
- Tag ID: Numeric identifier
- X: Centimeters from left edge
- Y: Centimeters from bottom edge
- Timestamp: Microseconds (UWB time)

### Tag JSON Format

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

### DXF Format

**Entities Used**:
- `LWPOLYLINE`: Court boundaries, areas
- `LINE`: Center line, etc.
- `CIRCLE`: Center circle, three-point arcs

**Example**:
```python
doc = ezdxf.readfile('court.dxf')
msp = doc.modelspace()

for entity in msp:
    if entity.dxftype() == 'LWPOLYLINE':
        points = entity.get_points()
        # [(x1, y1), (x2, y2), ...]
```

### Detection Cache Format

```json
{
  "frame_0": [
    {
      "bbox": [120.5, 200.3, 180.2, 350.7],
      "confidence": 0.87,
      "class_id": 0,
      "track_id": 1
    }
  ],
  "frame_5": [...]
}
```

---

## Mathematical Foundations

### Euclidean Distance

```
d = √[(x₂ - x₁)² + (y₂ - y₁)²]
```

**Properties**:
- Symmetric: d(A,B) = d(B,A)
- Triangle inequality: d(A,C) ≤ d(A,B) + d(B,C)
- Non-negative: d ≥ 0

### Homogeneous Coordinates

**2D Point**: (x, y)
**Homogeneous**: (x, y, 1) or (wx, wy, w) where w ≠ 0

**Convert back**:
```
(wx, wy, w) → (wx/w, wy/w)
```

**Advantages**:
- Represents transformations as matrix multiplication
- Handles projective transformations
- Infinity points: (x, y, 0)

### Golden Angle

```
φ = 137.5° ≈ 360° × (1 - 1/φ)
where φ = (1 + √5) / 2 (golden ratio)
```

**Use in tag colors**:
```javascript
hue = (tag_id × 137.5) % 360
color = hsl(hue, 70%, 50%)
```

**Result**: Maximally distinct colors for sequential IDs

### RANSAC Probability

**Probability of success**:
```
P(success) = 1 - (1 - wⁿ)ᵏ
```

Where:
- w: Inlier ratio (0-1)
- n: Points needed (4 for homography)
- k: Iterations

**Example**:
```
w = 0.7 (70% inliers)
n = 4
k = 2000

P(success) = 1 - (1 - 0.7⁴)²⁰⁰⁰
           ≈ 1.0 (virtually certain)
```

---

For API details, see **API_REFERENCE.md**.
For usage instructions, see **USER_GUIDE.md**.
For improvements, see **FUTURE_IMPROVEMENTS.md**.
