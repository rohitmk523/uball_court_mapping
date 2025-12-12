# Basketball Court Tracking - Working Scripts

This document contains ready-to-use commands for running the tracking pipeline with different configurations.

## Output Naming Convention

All output videos follow this naming pattern:
```
{video_name}_short_{start_time}_{end_time}_{quality}_bytetrack_{segmentation}.mp4
```

**Example:**
```
GX020018_1080p_short_00-26-00_00-27-00_large_bytetrack_sam.mp4
```

**Components:**
- `video_name`: Input video filename (e.g., `GX020018_1080p`)
- `short`: Indicates this is a segment of the full video
- `start_time`: Start time with dashes (e.g., `00-26-00` for 00:26:00)
- `end_time`: End time with dashes (e.g., `00-27-00` for 00:27:00)
- `quality`: SAM2 model quality (`tiny`, `small`, `base`, `large`)
- `bytetrack`: Tracker type (always ByteTrack)
- `segmentation`: `sam` if SAM2 enabled, `nosam` otherwise

## Prerequisites

```bash
# Activate conda environment
conda activate court_tracking

# Verify environment
python -c "import cv2, torch, ultralytics; print('✓ Environment ready')"
```

## SAM2 Test Script (`test_sam2_tracking.py`)

### Basic Commands

#### 1. Without SAM2 (YOLO + ByteTrack only)
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --no-display
```
**Output:** `GX010018_1080p_short_00-26-00_00-27-00_large_bytetrack_nosam.mp4`

**Features:**
- YOLO11 player detection
- ByteTrack multi-object tracking
- Bounding boxes visualization
- ~25-30 FPS on MPS

---

#### 2. With SAM2 (YOLO + SAM2 + ByteTrack)
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --no-display
```
**Output:** `GX010018_1080p_short_00-26-00_00-27-00_large_bytetrack_sam.mp4`

**Features:**
- YOLO11 player detection
- SAM2 precise segmentation
- ByteTrack with mask-based features
- Refined bounding boxes from masks
- ~1.4 FPS on MPS (large model)

---

#### 3. With SAM2 and Mask Visualization
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --show-masks \
  --no-display
```
**Output:** `GX010018_1080p_short_00-26-00_00-27-00_large_bytetrack_sam.mp4`

**What `--show-masks` does:**
- Displays colored segmentation masks as overlays
- Shows precise player outlines (pixel-accurate)
- Replaces bounding boxes with mask visualization
- Only works when `--sam2` is enabled

---

### Quality Settings

Choose SAM2 model quality based on your needs:

#### Tiny (Fastest, Good Quality)
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --quality tiny \
  --no-display
```
- Model: `sam2_hiera_tiny.pt` (~38 MB)
- Speed: ~5-8 FPS on MPS
- Use for: Quick testing, real-time applications

---

#### Small (Fast, Better Quality)
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --quality small \
  --no-display
```
- Model: `sam2_hiera_small.pt` (~155 MB)
- Speed: ~3-5 FPS on MPS
- Use for: Good balance of speed and quality

---

#### Base (Medium Speed, High Quality)
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --quality base \
  --no-display
```
- Model: `sam2_hiera_base_plus.pt` (~220 MB)
- Speed: ~2-3 FPS on MPS
- Use for: High-quality analysis

---

#### Large (Slowest, Best Quality) - **DEFAULT**
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --quality large \
  --no-display
```
- Model: `sam2_hiera_large.pt` (~224 MB)
- Speed: ~1.4 FPS on MPS
- Use for: Best quality, offline analysis

---

### Advanced Options

#### Custom Output Path
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --output my_custom_output.mp4 \
  --no-display
```

---

#### Live Display (Without Saving)
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:26:10 \
  --sam2 \
  --show-masks
```
Press `q` to quit during playback.

---

#### Force CPU/GPU Device
```bash
# Force CPU
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --device cpu \
  --no-display

# Force CUDA (NVIDIA GPU)
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --device cuda \
  --no-display

# Force MPS (Apple Silicon)
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --device mps \
  --no-display
```

---

#### Custom YOLO Model and Confidence
```bash
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --yolo-model yolo11x.pt \
  --confidence 0.6 \
  --no-display
```

---

## Full Tracking Pipeline

For the complete tracking pipeline with UWB tag association:

### 1. Calibration (One-time Setup)
```bash
# Start server
./start_server.sh

# Navigate to http://localhost:8000/calibration
# Load video frame and select correspondence points
```

### 2. Set Sync Point (One-time Setup)
```bash
# Navigate to http://localhost:8000/tracking
# Enter video frame and UWB timestamp for synchronization
```

### 3. Process Full Video
```bash
# Via web interface
# Navigate to http://localhost:8000/tracking
# Enter frame range and click "Process Video with YOLO"
```

---

## Performance Comparison

| Configuration | FPS (MPS) | Quality | Use Case |
|---------------|-----------|---------|----------|
| YOLO only | 25-30 | Good | Real-time tracking |
| YOLO + SAM2 (tiny) | 5-8 | Good | Balanced |
| YOLO + SAM2 (small) | 3-5 | Better | Quality focus |
| YOLO + SAM2 (base) | 2-3 | High | Analysis |
| YOLO + SAM2 (large) | 1.4 | Best | Offline analysis |

---

## Troubleshooting

### Script Not Found
```bash
# Ensure you're in the project root
cd /Users/rohitkale/Cellstrat/GitHub_Repositories/uball_court_mapping
```

### Missing SAM2 Models
```bash
# Download models
mkdir -p checkpoints

# Large model
curl -L -o checkpoints/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Base model
curl -L -o checkpoints/sam2_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Small model
curl -L -o checkpoints/sam2_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

# Tiny model
curl -L -o checkpoints/sam2_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

### Environment Not Activated
```bash
conda activate court_tracking
```

### Video File Not Found
```bash
# Verify video exists
ls -lh GX010018_1080p.MP4

# Or use absolute path
python test_sam2_tracking.py \
  --video /path/to/your/video.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --no-display
```

### Out of Memory (OOM)
```bash
# Use smaller SAM2 model
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --quality tiny \
  --no-display

# Or use CPU
python test_sam2_tracking.py \
  --video GX010018_1080p.MP4 \
  --start 00:26:00 \
  --end 00:27:00 \
  --sam2 \
  --device cpu \
  --no-display
```

---

## Quick Reference

### Get Help
```bash
python test_sam2_tracking.py --help
```

### Recommended Commands

**Quick Test (10 seconds, no SAM2):**
```bash
python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:26:10 --no-display
```

**Quality Analysis (1 minute, SAM2 large):**
```bash
python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2 --show-masks --no-display
```

**Fast Analysis (1 minute, SAM2 tiny):**
```bash
python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2 --quality tiny --no-display
```

---

## Next Steps

1. **Test the script:**
   ```bash
   python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:26:30 --sam2 --no-display
   ```

2. **Review output:**
   ```bash
   ls -lh GX010018_1080p_short_*.mp4
   ```

3. **Clean up old outputs:**
   ```bash
   rm -f test_output_*.mp4
   ```

4. **Process full game:**
   - Use web interface for full video processing
   - Apply calibration and UWB tag matching
   - Export results to JSON

---

**Last Updated:** December 12, 2025
**Version:** 2.0.0
**Status:** Production Ready
