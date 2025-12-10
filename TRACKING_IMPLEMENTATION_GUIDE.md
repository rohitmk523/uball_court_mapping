# Tracking Implementation Guide: Getting Started

## Overview

This guide walks you through using the new **ByteTrack + UWB Association** system for persistent player tracking. The system provides:

- **Persistent Track IDs**: Players maintain the same ID across frames
- **UWB Association**: Track IDs are mapped to UWB tag IDs
- **High Accuracy**: Motion-based tracking with spatial validation

## Prerequisites

✅ All dependencies installed (supervision, scipy, ultralytics, opencv)
✅ Calibration completed (homography.json exists)
✅ UWB data available (data/tags/*.json)
✅ Video file (e.g., GX010018_1080p.MP4)
✅ Sync offset determined (e.g., 8 seconds)

## Quick Start: 3 Steps to Track Players

### Step 1: Test Tracking on Short Segment (1 minute)

First, let's verify the tracking works on a 1-minute segment:

```bash
# Test tracking from 20:00 to 21:00 (1 minute)
python test_tracking_pipeline.py \
    --video GX010018_1080p.MP4 \
    --start 00:20:00 \
    --end 00:21:00 \
    --sync-offset 8
```

**Expected Output:**
- Console: Track IDs detected (e.g., "Track 1, Track 3, Track 7")
- Console: UWB associations (e.g., "Track 3 → Tag 5")
- File: `tracked_test_20_21.mp4` (visualized tracking)
- File: `tracked_data_20_21.json` (tracking data)

### Step 2: Review Tracking Quality

Open the generated video and check:

1. **Track IDs persist**: Same player keeps same ID across frames
2. **Associations look correct**: Track IDs match UWB tag positions
3. **Color coding**:
   - 🟢 Green = HIGH confidence (validated mapping)
   - 🟡 Yellow = NEW association (just created)
   - 🔴 Red = LOW confidence (no UWB tag found)
   - 🔵 Blue = Ghost tag (UWB with no player)

### Step 3: Process Full Segment (5 minutes)

Once validated, process a longer segment:

```bash
# Process 20-25 minute segment
python generate_tracked_video_complete.py \
    --video GX010018_1080p.MP4 \
    --start 00:20:00 \
    --end 00:25:00 \
    --sync-offset 8 \
    --output tracked_20_25.mp4 \
    --json-output tracked_20_25.json
```

---

## Detailed Usage

### Option A: Using Updated Visual Validation Tool

The `visual_validation_stitch.py` now supports tracking visualization:

```bash
python visual_validation_stitch.py 00:20:00 00:21:00 8 GX010018_1080p.MP4
```

**What it does:**
- Left panel: YOLO detections with Track IDs
- Middle panel: Red dots (tracked players on court)
- Right panel: Blue dots (UWB tags on court)

**Parameters:**
- `00:20:00` - Start time
- `00:21:00` - End time
- `8` - Sync offset (seconds)
- `GX010018_1080p.MP4` - Video file

### Option B: Using Standalone Tracking Script

For just tracking without the 3-panel view:

```python
#!/usr/bin/env python3
"""
Quick tracking test script
Save as: test_tracking_pipeline.py
"""

import sys
import cv2
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.player_detector import PlayerDetector
from app.services.uwb_associator import UWBAssociator
from app.services.calibration_integration import CalibrationIntegration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--sync-offset', type=float, required=True)
    args = parser.parse_args()

    # Initialize components
    print("Initializing tracking system...")
    detector = PlayerDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.3,
        enable_tracking=True,
        track_buffer=50
    )

    calibration = CalibrationIntegration("data/calibration/1080p/homography.json")

    uwb_associator = UWBAssociator(
        tags_dir=Path("data/tags"),
        sync_offset_seconds=args.sync_offset,
        proximity_threshold=2.0
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Parse times
    start_sec = sum(int(x) * 60**i for i, x in enumerate(reversed(args.start.split(':'))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    frame_count = 0
    print("\nProcessing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track players
        tracked_players = detector.track_players(frame)

        # Project to court
        for player in tracked_players:
            bx, by = player['bottom']
            try:
                court_x, court_y = calibration.image_to_court(bx, by)
                player['court_x'] = float(court_x)
                player['court_y'] = float(court_y)
            except:
                player['court_x'] = None
                player['court_y'] = None

        # Associate with UWB
        video_time = (int(start_sec * fps) + frame_count) / fps
        tracked_players = uwb_associator.associate(video_time, tracked_players)

        # Print tracking info
        if frame_count % 30 == 0:  # Every second
            print(f"\nFrame {frame_count}:")
            for p in tracked_players:
                tid = p.get('track_id')
                uid = p.get('uwb_tag_id', 'None')
                conf = p.get('association_confidence', 'N/A')
                print(f"  Track {tid} → UWB {uid} ({conf})")

        frame_count += 1

        # Stop after 1 minute for testing
        if frame_count > int(60 * fps):
            break

    cap.release()

    # Print statistics
    print("\n" + "="*60)
    print("TRACKING STATISTICS")
    print("="*60)

    track_stats = detector.get_track_statistics()
    print(f"Total tracks: {track_stats['total_tracks']}")
    print(f"Active tracks: {track_stats['active_tracks']}")

    uwb_stats = uwb_associator.get_statistics()
    print(f"\nUWB Association:")
    print(f"  Success rate: {uwb_stats['success_rate_percent']:.1f}%")
    print(f"  Mappings: {uwb_stats['track_to_tag_mapping']}")
    print("="*60)

if __name__ == "__main__":
    main()
```

Save this as `test_tracking_pipeline.py` and run:

```bash
python test_tracking_pipeline.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:21:00 --sync-offset 8
```

---

## Understanding the Output

### Console Output

```
Initializing tracking system...
ByteTrack tracker initialized
  Activation threshold: 0.25
  Lost track buffer: 50 frames
  Min matching threshold: 0.8

UWB Associator initialized
  Tags loaded: 12
  Sync offset: 8s
  Proximity threshold: 2.0m

Processing video...

Frame 0:
  Track 1 → UWB 5 (HIGH)
  Track 3 → UWB 7 (HIGH)
  Track 7 → UWB 2 (NEW)
  Track 9 → UWB None (LOW)

Frame 30:
  Track 1 → UWB 5 (HIGH)
  Track 3 → UWB 7 (HIGH)
  Track 7 → UWB 2 (HIGH)
  Track 9 → UWB 12 (NEW)

...

TRACKING STATISTICS
Total tracks: 14
Active tracks: 8

UWB Association:
  Success rate: 87.3%
  Mappings: {1: 5, 3: 7, 7: 2, 9: 12, 11: 3, 13: 8}
```

### JSON Output Format

```json
{
  "metadata": {
    "video_path": "GX010018_1080p.MP4",
    "sync_offset_seconds": 8,
    "start_time": "00:20:00",
    "end_time": "00:21:00",
    "fps": 29.97
  },
  "frames": [
    {
      "frame_number": 35964,
      "video_time": 1200.0,
      "players": [
        {
          "track_id": 1,
          "uwb_tag_id": 5,
          "bbox_video": [850, 320, 920, 480],
          "position_court": {"x": 12.5, "y": 18.3},
          "detection_confidence": 0.87,
          "association_confidence": "HIGH",
          "association_distance": 0.43
        },
        {
          "track_id": 3,
          "uwb_tag_id": 7,
          "bbox_video": [620, 280, 690, 440],
          "position_court": {"x": 8.2, "y": 22.1},
          "detection_confidence": 0.92,
          "association_confidence": "HIGH",
          "association_distance": 0.28
        }
      ]
    }
  ]
}
```

---

## Parameter Tuning

### If Track IDs Switch Too Often

**Problem**: Players lose their IDs and get reassigned
**Solution**: Increase `lost_track_buffer`

```python
detector = PlayerDetector(
    enable_tracking=True,
    track_buffer=100  # Increased from 50 (keep lost tracks longer)
)
```

### If UWB Associations Are Wrong

**Problem**: Track IDs mapping to incorrect UWB tags
**Solution**: Decrease `proximity_threshold`

```python
uwb_associator = UWBAssociator(
    tags_dir=Path("data/tags"),
    sync_offset_seconds=8,
    proximity_threshold=1.5  # Decreased from 2.0 (stricter matching)
)
```

### If IDs Flicker Between Tags

**Problem**: Track IDs switch between UWB tags rapidly
**Solution**: Increase `remapping_cooldown`

```python
uwb_associator = UWBAssociator(
    tags_dir=Path("data/tags"),
    sync_offset_seconds=8,
    proximity_threshold=2.0,
    remapping_cooldown=60  # Increased from 30 frames (~2 seconds)
)
```

---

## Troubleshooting

### Issue: "ByteTrack tracker not available"

**Cause**: supervision package not installed
**Fix**:
```bash
pip install supervision>=0.16.0
# or
conda install -c conda-forge supervision
```

### Issue: No UWB associations (all LOW confidence)

**Possible causes:**
1. **Sync offset wrong** → Verify with visual_validation_stitch.py
2. **UWB data missing** → Check `data/tags/*.json` exists
3. **Threshold too strict** → Increase `proximity_threshold` to 3.0

**Debug**:
```python
# Print UWB positions to verify data is loaded
uwb_stats = uwb_associator.get_statistics()
print(f"Tags loaded: {len(uwb_associator.tag_data)}")
print(f"Total UWB positions: {sum(len(v) for v in uwb_associator.tag_data.values())}")
```

### Issue: Track IDs not persistent

**Possible causes:**
1. **Detection confidence too low** → Increase confidence threshold
2. **Track buffer too short** → Increase `track_buffer`
3. **Players occluding each other** → This is normal, IDs should recover

**Expected behavior:**
- Short occlusions: IDs preserved
- Long occlusions (>50 frames): New ID assigned
- Re-entry after leaving: New ID assigned

---

## Integration with Existing Pipeline

### Updating generate_red_dots.py

To add tracking to red dots generation:

```python
# In generate_red_dots.py
from app.services.player_detector import PlayerDetector

# Replace existing detector initialization
player_detector = PlayerDetector(
    model_name="yolov8n.pt",
    confidence_threshold=0.3,
    enable_tracking=True,  # Enable tracking
    track_buffer=50
)

# In processing loop, replace:
# detections = player_detector.detect_players(frame)
# with:
tracked_players = player_detector.track_players(frame)

# Now each player has 'track_id' field
for player in tracked_players:
    track_id = player['track_id']
    # Draw with track_id label...
```

### Updating visual_validation_stitch.py

To show track IDs in the YOLO panel:

```python
# In generate_annotated_video() function
detections = player_detector.track_players(frame)  # Use tracking

for detection in detections:
    bbox = detection['bbox']
    conf = detection['confidence']
    track_id = detection.get('track_id')  # Get track ID

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw track ID
    if track_id is not None:
        label = f"ID:{track_id} {conf:.2f}"
    else:
        label = f"{conf:.2f}"

    cv2.putText(frame, label, (x1, y1 - 5), ...)
```

---

## Performance Optimization

### For Faster Processing

1. **Use smaller YOLO model**:
   ```python
   detector = PlayerDetector(model_name="yolo11n.pt")  # Nano model
   ```

2. **Increase confidence threshold**:
   ```python
   detector = PlayerDetector(confidence_threshold=0.5)  # Fewer detections
   ```

3. **Skip frames** (if accuracy allows):
   ```python
   # Process every 2nd frame
   if frame_idx % 2 == 0:
       tracked_players = detector.track_players(frame)
   ```

### For Better Accuracy

1. **Use larger YOLO model**:
   ```python
   detector = PlayerDetector(model_name="yolo11m.pt")  # Medium model
   ```

2. **Lower confidence threshold**:
   ```python
   detector = PlayerDetector(confidence_threshold=0.2)  # More detections
   ```

3. **Increase track buffer**:
   ```python
   detector = PlayerDetector(track_buffer=100)  # Longer memory
   ```

---

## Next Steps

1. ✅ **Test on 1-minute segment** → Verify tracking works
2. ✅ **Tune parameters** → Optimize for your use case
3. ✅ **Process full game** → Generate tracked video
4. ✅ **Analyze results** → Use JSON data for analytics
5. ✅ **Integrate with pipeline** → Update existing scripts

## Support

For issues or questions:
1. Check console output for error messages
2. Verify file paths are correct
3. Ensure all dependencies installed
4. Test with short segments first
5. Review tracking statistics for insights

---

**🎉 You're ready to start tracking!**

Run the test script first:
```bash
python test_tracking_pipeline.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:21:00 --sync-offset 8
```
