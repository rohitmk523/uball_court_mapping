# User Guide

Complete step-by-step guide for using the Basketball Court Tracking System.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Workflow 1: Visualizing UWB Tag Data](#workflow-1-visualizing-uwb-tag-data)
3. [Workflow 2: Calibrating Video with Court](#workflow-2-calibrating-video-with-court)
4. [Workflow 3: Tracking and Matching](#workflow-3-tracking-and-matching)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

### Prerequisites

Before starting, ensure you have:

1. ✅ Conda environment created and activated
2. ✅ All dependencies installed
3. ✅ Data files in place:
   - `logs/session_1763778442483.log` - UWB log file
   - `dxf/court_2.dxf` - Court DXF file
   - `GX020018.MP4` - Video file

### Starting the Server

**Option 1: Using the startup script**
```bash
./start_server.sh
```

**Option 2: Manual start**
```bash
conda activate court_tracking
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify server is running**:
- Open browser: http://localhost:8000
- You should see the Tag Visualization page

---

## Workflow 1: Visualizing UWB Tag Data

This workflow shows you how to visualize UWB tag positions and animate their movement across the basketball court.

### Step 1: Access the Tag Visualization Page

1. Open your browser
2. Navigate to: **http://localhost:8000/**
3. You'll see the main page with:
   - Control panel on the left
   - Court visualization canvas on the right

### Step 2: Process the UWB Log File

The UWB log file contains raw position data that needs to be parsed.

1. **Click "Process Log File" button** in the control panel
2. **Wait for processing** (2-5 seconds)
3. **Status message** will appear showing:
   ```
   Successfully processed 19 tags with 1,384,058 total positions
   ```

**What happens behind the scenes**:
- Log file is read line by line
- Each line is parsed: `Tag ID | X | Y | Timestamp`
- Positions are grouped by tag ID
- Individual JSON files are created in `data/output/tags/`
- 19 tags are processed with timestamps ranging from 301564 to 268757500

### Step 3: View the Court

Once processing completes:

1. **Court appears** on the canvas
2. **Court dimensions**: 2460cm × 1730cm (basketball court size)
3. **All 19 tags** are rendered on the court
4. **Each tag** has a unique color (using golden angle: 137.5°)

### Step 4: Control the Animation

#### Playback Controls

**Play Button**:
- Click to start animation
- Tags move according to their recorded positions
- Animation loops when it reaches the end

**Pause Button**:
- Click to pause at current timestamp
- Tags remain at their current positions

**Reset Button**:
- Returns to the beginning (earliest timestamp)
- All tags return to starting positions

#### FPS Control (Animation Speed)

1. **Use the FPS slider** to adjust speed (1-30 FPS)
2. **Lower FPS** (1-5):
   - Slower animation
   - Better for detailed observation
   - Good for analyzing specific movements

3. **Medium FPS** (10-15):
   - Balanced speed
   - Good for general viewing
   - Default setting

4. **Higher FPS** (20-30):
   - Faster animation
   - Quick overview of movement patterns
   - More fluid motion

**Current FPS Display**:
- Shows actual FPS being achieved
- May differ from target if system is slow

#### Timeline Scrubbing

1. **Use the timeline slider** to jump to any timestamp
2. **Drag the slider** to scrub through time
3. **Current timestamp** is displayed below
4. **Tags update** to show positions at that moment

### Step 5: Monitor Active Tags

**Active Tags Display**:
- Shows number of tags currently visible
- Tag list shows all tag IDs with their current positions
- Scroll to see all tags if list is long

**Tag Information**:
```
Tag 1587672: (169.0, 468.0)
Tag 1587036: (523.0, 892.0)
...
```

### Understanding the Visualization

**Coordinate System**:
- Origin (0,0) is at bottom-left of court
- X-axis: Horizontal (0 to 2460cm)
- Y-axis: Vertical (0 to 1730cm)
- Units: Centimeters

**Tag Colors**:
- Each tag has a unique color
- Colors are distributed using golden angle (137.5°)
- Same tag always has same color
- Makes it easy to track individual players

**Court Elements**:
- Outer boundary (black)
- Three-point lines
- Free throw areas (key)
- Center circle
- Half-court line

---

## Workflow 2: Calibrating Video with Court

Calibration creates a mapping between court coordinates (from UWB) and video coordinates (from camera). This is essential for matching tags to players in the video.

### Understanding Calibration

**What is Homography?**
- A 3×3 matrix that transforms points between two coordinate systems
- Allows converting court position (X, Y in cm) to video pixel (x, y)
- Computed from correspondence points you select

**Why is it needed?**
- UWB tags give positions in court coordinates (meters)
- Video shows the court from camera perspective (pixels)
- We need to map between these two systems
- Enables matching tags to detected players

### Step 1: Access Calibration Page

1. Navigate to: **http://localhost:8000/calibration**
2. You'll see:
   - Instructions at the top
   - Two canvas panels side by side:
     - **Left**: Court image (from DXF)
     - **Right**: Video frame (to be loaded)
   - Control buttons at the bottom

### Step 2: Load a Video Frame

1. **Choose a frame number**:
   - Enter frame number in the input field
   - Recommendation: Choose a frame where:
     - Court is clearly visible
     - Court lines are sharp and well-lit
     - No occlusions
     - Example: Frame 100-500

2. **Click "Load Video Frame"**:
   - Video frame will appear on the right canvas
   - May take a moment to load (video is 11GB)

**Troubleshooting**:
- If frame doesn't load, check:
  - Video file exists: `ls -lh GX020018.MP4`
  - Frame number is within video range
  - Server has read permissions

### Step 3: Select Correspondence Points

This is the most important step! You need to click corresponding points on both images.

#### Point Selection Process

**Alternating Pattern**:
1. Click a point on **Court** (left)
2. Then click the **same point** on **Video** (right)
3. Repeat for 10-15 points

**Visual Feedback**:
- Numbered markers appear on both canvases
- Court canvas gets blue highlight (click here first)
- Video canvas gets green highlight (click here second)
- Point count updates in status

**If you click in wrong order**:
- Alert appears: "Please click on the video frame first!"
- Or: "Please click on the court first!"

#### Choosing Good Points

**Best Points** (use these):
- ✅ Court line intersections
- ✅ Three-point line corners
- ✅ Free throw line endpoints
- ✅ Center court circle points
- ✅ Baseline corners
- ✅ Key area corners

**Avoid**:
- ❌ Points on players or objects
- ❌ Occluded or unclear points
- ❌ Points outside the court
- ❌ Shadows or reflections

**Distribution**:
- Spread points evenly across the court
- Include points from all four corners
- Include center points
- More points near areas of interest

#### Example Point Selection

```
Point 1: Top-left corner of court boundary
Point 2: Top-right corner of court boundary
Point 3: Bottom-left corner of court boundary
Point 4: Bottom-right corner of court boundary
Point 5: Center court circle (top)
Point 6: Center court circle (bottom)
Point 7: Left three-point line corner (top)
Point 8: Right three-point line corner (top)
Point 9: Left free throw line endpoint
Point 10: Right free throw line endpoint
Point 11: Half-court line intersection with sideline (left)
```

### Step 4: Verify Point Selection

1. **Check point count**: Should show "Points selected: 10" (minimum)
2. **Visually inspect**: Markers should be at corresponding locations
3. **Use "Undo Last Point"** if you made a mistake:
   - Removes the most recent point pair
   - Can undo multiple times

4. **Use "Clear All Points"** to start over if needed

### Step 5: Submit Calibration

1. **Click "Submit Calibration"** button
2. **Wait for processing** (<1 second)
3. **Success message** appears:
   ```
   Calibration successful! Homography matrix computed.
   ```

**What happens**:
- Court points and video points are sent to backend
- OpenCV computes homography matrix using RANSAC
- RANSAC removes outliers (up to 5 pixels error)
- Matrix is saved to `data/calibration/calibration.json`
- You can now proceed to tracking!

### Step 6: Verify Calibration

**Check calibration status**:
- Status shows: "Calibrated"
- Point count shows: 11 (or however many you used)
- Timestamp shows when calibration was done

**Re-calibrate if needed**:
- Simply select new points and submit again
- Previous calibration is overwritten

**Test calibration quality**:
- Proceed to tracking workflow
- Check if tag positions align well with players
- If alignment is poor, re-calibrate with better points

---

## Workflow 3: Tracking and Matching

This workflow processes the video with YOLO, tracks players, and matches them with UWB tags.

### Overview

**Three main steps**:
1. **Synchronization**: Align video timeline with UWB timeline
2. **Video Processing**: Detect and track players with YOLO
3. **Playback**: View matched tags on players

### Step 1: Access Tracking Page

1. Navigate to: **http://localhost:8000/tracking**
2. You'll see three sections:
   - **Section 1**: Synchronization
   - **Section 2**: Video Processing
   - **Section 3**: Playback

### Step 2: Set Synchronization Point

**Why sync?**
- Video has frame numbers (0, 1, 2, ...)
- UWB has timestamps (301564, 301565, ...)
- We need to map frame numbers to UWB timestamps

**Finding a sync point**:
1. Find an **event visible in both systems**:
   - Player enters court
   - Ball possession change
   - Player at specific location
   - Coach signal or gesture

2. **Determine frame number**:
   - Use video player to find the frame
   - Note the frame number
   - Example: Frame 100

3. **Determine UWB timestamp**:
   - Look at UWB data around that time
   - Find corresponding timestamp
   - Example: 301564

**Enter sync point**:
1. Enter **Video Frame**: `100`
2. Enter **UWB Timestamp**: `301564`
3. Click **"Set Sync Point"**
4. Status shows: "✓ Synced: Frame 100 = Timestamp 301,564"

**Sync saved**: `data/calibration/sync_point.json`

**Timestamp calculation**:
```
For any frame F:
uwb_timestamp = sync_uwb + (F - sync_frame) × (1,000,000 / video_fps)

Example with 30 FPS:
Frame 150: 301564 + (150-100) × (1,000,000/30) = 1,969,897
```

### Step 3: Configure Video Processing

**Processing Parameters**:

1. **Start Frame**:
   - First frame to process
   - Default: 0
   - Example: 0 (beginning of video)

2. **End Frame**:
   - Last frame to process
   - Default: Last frame of video
   - Example: 500 (for testing)
   - **Recommendation**: Start with 500 frames for testing

3. **Frame Skip**:
   - Process every Nth frame
   - Default: 5
   - Higher = faster but less temporal resolution
   - **Recommendation**: Use 5 for testing, 1-3 for final analysis

4. **Confidence Threshold** (optional):
   - YOLO detection confidence
   - Default: 0.5
   - Higher = fewer false positives but might miss players
   - Lower = more detections but more false positives

**Frame Skip Strategy**:

| Skip | Frames Processed | Time | Use Case |
|------|------------------|------|----------|
| 1 | All frames | Slowest | Final analysis |
| 3 | Every 3rd | Moderate | Good tracking |
| 5 | Every 5th | Fast | Testing |
| 10 | Every 10th | Very fast | Quick test |
| 30 | Every 30th (1 per sec) | Fastest | Initial setup |

**Example Configuration**:
```
Start Frame: 0
End Frame: 500
Frame Skip: 5
→ Processes frames: 0, 5, 10, 15, ..., 495
→ Total: 100 frames
→ Time: ~2-5 minutes
```

### Step 4: Process Video with YOLO

1. **Click "Process Video with YOLO"**
2. **Loading overlay appears** with:
   - Spinner animation
   - "Processing video with YOLO..."
   - "Processing ~100 frames. This may take several minutes."

3. **Processing happens**:
   - YOLOv11 detects persons in each frame
   - ByteTrack assigns consistent track IDs
   - Results are cached to avoid reprocessing
   - Tag-to-player matching is computed

4. **Progress monitoring**:
   - Watch server terminal for progress logs
   - Processing time depends on:
     - Number of frames
     - Hardware (GPU vs CPU)
     - Frame skip value

**Processing Time Estimates**:

| Frames | Skip | Actual Frames | GPU Time | CPU Time |
|--------|------|---------------|----------|----------|
| 500 | 5 | 100 | 2-3 min | 5-10 min |
| 500 | 1 | 500 | 8-12 min | 30-45 min |
| 5000 | 5 | 1000 | 15-25 min | 2-4 hours |

5. **Completion**:
   - Loading overlay disappears
   - Success message appears:
     ```
     ✓ Processed 100 frames. Found 523 detections (avg 5.2 per frame)
     ```
   - Playback controls are enabled

**What's cached**:
- YOLO detections: `data/cache/detections/detections_0_500_5.json`
- To reprocess: Delete cache file or use different parameters

### Step 5: Playback and Analysis

Once processing completes, you can play back the results.

#### Playback Controls

1. **Play Button**:
   - Starts frame-by-frame playback
   - Updates at ~10 FPS (for smooth visualization)
   - Shows matched tags on court

2. **Pause Button**:
   - Pauses at current frame
   - Allows inspection of current state

3. **Reset Button**:
   - Returns to first frame
   - Resets all displays

4. **Timeline Scrubber**:
   - Drag to jump to specific frame
   - Shows frame number as you drag

#### Understanding the Display

**Statistics Panel**:
```
Current Frame: 150 / 100
Current Timestamp: 1,969,897
Players Detected: 5
Tags Matched: 3
```

**Court Visualization**:
- Shows all tags at current timestamp
- Matched tags are highlighted
- Tag positions from UWB data
- Court is rendered from DXF

**Video Panel** (future feature):
- Will show video frame with bounding boxes
- Track IDs displayed
- Matched tags shown

#### Interpreting Results

**Good Match**:
```json
{
  "tag_id": 1587672,
  "matched_player": {
    "track_id": 1,
    "distance_cm": 45.2,  // < 200cm
    "video_position": [150, 350],
    "court_position": [520, 890]
  }
}
```
- Distance < 200cm (threshold)
- Tag and player are close
- Match is reliable

**Unmatched Tag**:
```json
{
  "tag_id": 1587036,
  "matched_player": null
}
```
- Reasons:
  - Player too far from tag (> 200cm)
  - Player not detected in video
  - Tag outside camera view

**Unmatched Player**:
```json
{
  "track_id": 3,
  "reason": "No tag within 200cm threshold"
}
```
- Reasons:
  - Player not wearing a tag
  - Tag data not available at this time
  - Calibration error (wrong court position)

### Step 6: Analyzing the Data

**Frame-by-Frame Analysis**:
1. Pause at interesting frames
2. Check matched tags
3. Verify distances are reasonable
4. Note any mismatches

**Quality Checks**:
- Do matched positions make sense?
- Are distances consistently small (< 100cm)?
- Do track IDs remain consistent?
- Are tags switching between players?

**If quality is poor**:
1. **Re-calibrate** with better points
2. **Adjust detection threshold** (try 0.6 or 0.7)
3. **Use smaller frame skip** (1 or 3)
4. **Check sync point** (verify it's accurate)

---

## Advanced Features

### Custom Processing Parameters

**Adjust confidence threshold**:
```bash
# Via API
curl -X POST "http://localhost:8000/api/tracking/process?conf_threshold=0.7"
```

**Process specific range**:
```bash
# Frames 1000-1500
curl -X POST "http://localhost:8000/api/tracking/process?start_frame=1000&end_frame=1500&frame_skip=3"
```

### Batch Processing

Process video in chunks for better memory management:

```bash
# Process 0-1000
curl -X POST "http://localhost:8000/api/tracking/process?start_frame=0&end_frame=1000&frame_skip=5"

# Process 1000-2000
curl -X POST "http://localhost:8000/api/tracking/process?start_frame=1000&end_frame=2000&frame_skip=5"

# Continue...
```

### Exporting Data

**Export matched tags**:
```bash
# Get matches for frame 150
curl http://localhost:8000/api/tracking/matched/150 > frame_150_matches.json
```

**Export all tag data**:
```bash
# Copy tag JSON files
cp -r data/output/tags/ exported_tags/
```

### Using Different YOLO Models

Edit `app/services/detector.py`:
```python
# Change model
def get_detector(model_name='yolo11s.pt'):  # Use small instead of nano
    # 'yolo11n.pt' - Nano (fastest, least accurate)
    # 'yolo11s.pt' - Small
    # 'yolo11m.pt' - Medium
    # 'yolo11l.pt' - Large
    # 'yolo11x.pt' - XLarge (slowest, most accurate)
```

### Adjusting Match Threshold

Edit `app/core/config.py`:
```python
# Change from 200cm to 150cm for stricter matching
TAG_MATCH_THRESHOLD_CM = 150
```

Restart server after changes.

---

## Troubleshooting

### Server Issues

**Problem**: Server won't start

**Solutions**:
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Check conda environment
conda env list
conda activate court_tracking

# Verify dependencies
python -c "import fastapi, cv2, ultralytics; print('OK')"
```

**Problem**: Server crashes during processing

**Solutions**:
- Check server logs for errors
- Verify video file is accessible
- Ensure enough disk space (cache files)
- Try processing fewer frames

### Tag Visualization Issues

**Problem**: Tags not appearing after processing

**Solutions**:
```bash
# Check if JSON files were created
ls data/output/tags/
# Should show 19 .json files

# Check one file
cat data/output/tags/1587672.json | head -20

# Reprocess if needed
curl -X POST http://localhost:8000/api/tags/process
```

**Problem**: Court not rendering

**Solutions**:
```bash
# Check court image exists
ls -lh data/calibration/court_image.png

# Regenerate if needed
curl -X POST http://localhost:8000/api/court/regenerate-image

# Check DXF file
ls -lh dxf/court_2.dxf
```

### Calibration Issues

**Problem**: "Calibration Required" error

**Solutions**:
- Complete calibration workflow first
- Check file exists: `ls data/calibration/calibration.json`
- Re-run calibration if file is missing

**Problem**: Poor alignment after calibration

**Solutions**:
1. **Use more points** (15 instead of 10)
2. **Better distribute points** across court
3. **Use clearer points** (line intersections)
4. **Check for mistakes** in point selection
5. **Try different video frame** (better lighting)

**Problem**: Can't load video frame

**Solutions**:
```bash
# Check video file
ls -lh GX020018.MP4
file GX020018.MP4

# Test with Python
python -c "import cv2; cap = cv2.VideoCapture('GX020018.MP4'); print(cap.isOpened())"
```

### Video Processing Issues

**Problem**: "Sync Point Required" error

**Solutions**:
- Set sync point before processing
- Check file: `ls data/calibration/sync_point.json`
- Re-set sync point if missing

**Problem**: Processing is very slow

**Solutions**:
1. **Use higher frame skip** (10 or 30)
2. **Process fewer frames** (100-200)
3. **Check GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
4. **Close other applications**
5. **Use smaller YOLO model** (nano)

**Problem**: Out of memory errors

**Solutions**:
- Increase frame skip
- Process in smaller batches
- Clear detection cache: `rm -rf data/cache/detections/*`
- Restart server
- Close other applications

**Problem**: No players detected

**Solutions**:
1. **Lower confidence threshold** (0.3 or 0.4)
2. **Check video frame** (are players visible?)
3. **Verify YOLO model loaded**
4. **Check frame range** (right part of video?)

### Matching Issues

**Problem**: No tags matched to players

**Solutions**:
1. **Check calibration quality**
2. **Verify sync point is accurate**
3. **Check timestamp range** (tags available?)
4. **Increase match threshold** (300cm temporarily)
5. **Verify camera orientation** (left court = top video?)

**Problem**: Wrong tags matched

**Solutions**:
1. **Re-calibrate** with more accurate points
2. **Verify sync point** (check specific event)
3. **Check for calibration drift**
4. **Reduce match threshold** (150cm)

---

## Tips and Best Practices

### Calibration Tips

**Best Practices**:
1. ✅ Use 12-15 points (more is better)
2. ✅ Distribute evenly across court
3. ✅ Include all four corners
4. ✅ Use sharp, clear points
5. ✅ Double-check each point before clicking
6. ✅ Use court line intersections
7. ✅ Test with different frames if needed

**Common Mistakes**:
1. ❌ Too few points (< 10)
2. ❌ Points clustered in one area
3. ❌ Using blurry or occluded points
4. ❌ Clicking in wrong order (court → video)
5. ❌ Points on moving objects
6. ❌ Not verifying accuracy

### Video Processing Tips

**Optimal Settings**:
```
Initial Testing:
- Start: 0
- End: 500
- Skip: 5
- Confidence: 0.5

Fine-Tuned:
- Start: 0
- End: 5000
- Skip: 3
- Confidence: 0.6

Final Analysis:
- Start: 0
- End: All
- Skip: 1
- Confidence: 0.7
```

**Performance Tips**:
1. Start with small ranges for testing
2. Use frame skip during development
3. Enable GPU if available
4. Cache results for reuse
5. Process during off-hours for long videos

### Sync Point Tips

**Finding Good Sync Points**:
1. **Clear events**:
   - Ball goes out of bounds
   - Timeout called
   - Player substitution
   - Free throw

2. **Verification**:
   - Note exact frame number
   - Find same event in UWB data
   - Look for corresponding movement pattern
   - Verify timestamp makes sense

3. **Multiple sync points**:
   - Test different sync points
   - Compare results
   - Use the one with best matches

### Data Quality Tips

**Verify Quality**:
1. Check processing statistics
2. Review matched tags
3. Look for consistent track IDs
4. Verify distances are reasonable
5. Spot-check random frames

**Improve Quality**:
1. Better calibration (more points)
2. Accurate sync point
3. Appropriate frame skip
4. Optimal confidence threshold
5. Clean input data

---

## Keyboard Shortcuts

Currently, the system uses mouse/touch interaction. Future versions may add:

- `Space`: Play/Pause
- `R`: Reset
- `←/→`: Previous/Next frame
- `[/]`: Decrease/Increase FPS
- `U`: Undo last calibration point
- `C`: Clear all points

---

## Data Backup

**Important files to backup**:
```bash
# Calibration data
cp data/calibration/calibration.json backup/
cp data/calibration/sync_point.json backup/

# Processed tags
cp -r data/output/tags/ backup/

# Detection cache (if want to keep)
cp -r data/cache/detections/ backup/
```

**Restore from backup**:
```bash
cp backup/calibration.json data/calibration/
cp backup/sync_point.json data/calibration/
cp -r backup/tags/* data/output/tags/
```

---

## Next Steps

After completing all workflows:

1. **Export results** for analysis
2. **Create visualizations** from matched data
3. **Analyze movement patterns**
4. **Generate statistics**
5. **Share findings** with team

For technical details, see **TECHNICAL_DETAILS.md**.
For API usage, see **API_REFERENCE.md**.
For future improvements, see **FUTURE_IMPROVEMENTS.md**.
