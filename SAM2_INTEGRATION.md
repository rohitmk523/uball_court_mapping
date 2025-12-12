# SAM2 Integration Guide

## Overview

This project now supports **SAM2 (Segment Anything Model 2)** for enhanced player segmentation and tracking. SAM2 works in conjunction with YOLO detection to provide:

- **Better bounding boxes**: Refined, tighter bounding boxes based on precise segmentation masks
- **Segmentation masks**: Pixel-accurate player masks for visualization and analysis
- **Centroid calculation**: More accurate position estimation using mask centroids
- **Occlusion handling**: Improved tracking in crowded scenes with overlapping players

## Installation

### 1. Install SAM2

```bash
# Install SAM2 from GitHub
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. Install Dependencies

```bash
# Install additional dependencies
pip install torch>=2.0.0 torchvision>=0.15.0 hydra-core>=1.3.0 iopath>=0.1.10
```

### 3. Download SAM2 Checkpoints

Download a SAM2 model checkpoint from the [official repository](https://github.com/facebookresearch/segment-anything-2):

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SAM2 Hiera-Large model (recommended for quality)
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Or SAM2 Hiera-Base (faster, smaller)
# wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

## Configuration

### Environment Variable

Enable/disable SAM2 using the `USE_SAM2` environment variable:

```bash
# Enable SAM2
export USE_SAM2=true

# Disable SAM2
export USE_SAM2=false
```

### Configuration File

Edit `config/sam2_config.json` to customize SAM2 settings:

```json
{
  "sam2": {
    "enabled": false,
    "model_cfg": "sam2_hiera_l.yaml",
    "checkpoint_path": "checkpoints/sam2_hiera_large.pt",
    "device": "cpu",
    "min_mask_area": 1000,
    "stability_score_thresh": 0.88,
    "box_expansion": 10
  },
  "visualization": {
    "show_masks": true,
    "mask_alpha": 0.5,
    "mask_only": false,
    "show_centroids": true,
    "show_refined_bbox": true
  },
  "tracking": {
    "use_mask_features": true,
    "use_mask_centroid": true,
    "use_refined_bbox": true
  }
}
```

### Configuration Options

#### SAM2 Settings
- **`enabled`**: Enable/disable SAM2 (overridden by `USE_SAM2` env var)
- **`model_cfg`**: SAM2 model configuration (e.g., `sam2_hiera_l.yaml`, `sam2_hiera_b.yaml`)
- **`checkpoint_path`**: Path to SAM2 checkpoint file
- **`device`**: Device for inference (`cpu`, `cuda`, or `mps`)
- **`min_mask_area`**: Minimum mask area in pixels (filters out small/noisy masks)
- **`stability_score_thresh`**: Stability score threshold (0-1, higher = stricter)
- **`box_expansion`**: Pixels to expand YOLO bbox for SAM2 prompting

#### Visualization Settings
- **`show_masks`**: Display segmentation masks
- **`mask_alpha`**: Mask overlay transparency (0-1)
- **`mask_only`**: Show only masks without video background
- **`show_centroids`**: Display mask centroids
- **`show_refined_bbox`**: Display refined bounding boxes

#### Tracking Settings
- **`use_mask_features`**: Use mask features for tracking
- **`use_mask_centroid`**: Use mask centroid for position (instead of bbox center)
- **`use_refined_bbox`**: Use refined bbox from mask (instead of YOLO bbox)

## Usage

### Basic Example

```python
from app.services.player_detector import PlayerDetector
import cv2

# Initialize detector with SAM2
detector = PlayerDetector(
    model_name="yolo11n.pt",
    confidence_threshold=0.5,
    device="cpu",
    enable_tracking=True,
    use_sam2=True  # Enable SAM2
)

# Load frame
frame = cv2.imread("frame.jpg")

# Detect and track players with SAM2
tracked_players = detector.track_players(frame)

# Each tracked player now has:
# - track_id: Persistent tracking ID
# - bbox: Bounding box
# - mask: Segmentation mask (if SAM2 enabled)
# - refined_bbox: Tighter bbox from mask
# - mask_centroid: (cx, cy) mask centroid
# - mask_area: Mask area in pixels
# - mask_confidence: SAM2 stability score

# Visualize with masks
output_frame = detector.draw_detections(
    frame,
    tracked_players,
    show_masks=True
)

cv2.imshow("Output", output_frame)
cv2.waitKey(0)
```

### Using Environment Variable

```bash
# Run with SAM2 enabled
USE_SAM2=true python test_tracking_pipeline.py

# Run without SAM2
USE_SAM2=false python test_tracking_pipeline.py
```

### Programmatic Override

```python
# Override config and env var
detector = PlayerDetector(
    model_name="yolo11n.pt",
    use_sam2=True  # Force enable
)

# Or force disable
detector = PlayerDetector(
    model_name="yolo11n.pt",
    use_sam2=False  # Force disable
)
```

## Performance Considerations

### Device Selection

- **CPU**: Slowest, ~1-2 FPS with SAM2
- **CUDA (NVIDIA GPU)**: Fast, ~10-15 FPS with SAM2
- **MPS (Apple Silicon)**: Medium, ~5-8 FPS with SAM2

### Model Selection

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `sam2_hiera_tiny` | ~38 MB | Fast | Good |
| `sam2_hiera_small` | ~155 MB | Medium | Better |
| `sam2_hiera_base_plus` | ~220 MB | Medium | Better |
| `sam2_hiera_large` | ~224 MB | Slow | Best |

### Optimization Tips

1. **Use GPU**: Set `device="cuda"` or `device="mps"` in config
2. **Lower stability threshold**: Reduce `stability_score_thresh` to 0.80-0.85
3. **Increase min area**: Set `min_mask_area=2000` to filter small detections
4. **Reduce box expansion**: Set `box_expansion=5` for faster processing

## Architecture

### Pipeline Flow

```
Frame → YOLO Detection → SAM2 Segmentation → ByteTrack Tracking → Output
         ↓                  ↓                   ↓
      Bounding Boxes    Precise Masks      Persistent IDs
                           ↓
                    Refined Positions
```

### Component Integration

1. **PlayerDetector**: Main entry point
   - Loads YOLO model
   - Optionally loads SAM2 model
   - Manages tracker

2. **SAM2Segmenter**: Segmentation service
   - Receives YOLO detections as prompts
   - Generates precise masks
   - Refines bounding boxes

3. **ByteTrackTracker**: Tracking engine
   - Maintains persistent IDs
   - Optionally uses mask features
   - Handles occlusions

## Troubleshooting

### SAM2 Not Available

```
WARNING: SAM2 not available. Install from: https://github.com/facebookresearch/segment-anything-2
```

**Solution**: Install SAM2 using pip:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Checkpoint Not Found

```
ERROR: Failed to load SAM2 model: [Errno 2] No such file or directory: 'checkpoints/sam2_hiera_large.pt'
```

**Solution**: Download the checkpoint:
```bash
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### Out of Memory

```
ERROR: CUDA out of memory
```

**Solution**:
1. Use smaller model: `sam2_hiera_base_plus` or `sam2_hiera_small`
2. Process fewer detections per frame
3. Use CPU instead of GPU (slower but no memory issues)

### Slow Performance

**Solution**:
1. Use GPU: Set `device="cuda"` or `device="mps"`
2. Use smaller model: `sam2_hiera_small`
3. Reduce video resolution
4. Process every Nth frame instead of every frame

## Examples

### Example 1: Process Video Segment

```python
import cv2
from app.services.player_detector import PlayerDetector

# Initialize with SAM2
detector = PlayerDetector(
    model_name="yolo11n.pt",
    enable_tracking=True,
    use_sam2=True,
    device="mps"  # or "cuda" or "cpu"
)

# Open video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Track with SAM2
    tracked_players = detector.track_players(frame)

    # Visualize
    output = detector.draw_detections(frame, tracked_players, show_masks=True)

    cv2.imshow("SAM2 Tracking", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Compare With/Without SAM2

```bash
# Without SAM2
USE_SAM2=false python test_tracking_pipeline.py --video test.mp4 --output output_yolo.mp4

# With SAM2
USE_SAM2=true python test_tracking_pipeline.py --video test.mp4 --output output_sam2.mp4
```

## References

- [SAM2 Official Repository](https://github.com/facebookresearch/segment-anything-2)
- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Supervision](https://github.com/roboflow/supervision)
