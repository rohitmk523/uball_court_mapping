# Roboflow Dataset Setup Guide

## 1. Dataset Annotation (In Roboflow UI)

### Classes to Annotate (4 classes):

1. **players** (class 0)
   - Bounding box around each player
   - Include entire body (head to feet)
   - Label all players in frame (even partially visible)

2. **ball** (class 1)
   - Bounding box around basketball
   - Small tight box
   - Label even when held by player

3. **rim** (class 2)
   - Bounding box around basketball rim
   - Include entire hoop structure
   - Usually 1-2 rims per frame (depending on camera view)

4. **court_keypoints** (class 3)
   - Bounding box around court line intersections
   - Annotate ~10-15 keypoints per frame:
     - Corner intersections (4)
     - Free-throw line intersections (4)
     - Center court intersections (3)
     - 3-point line intersections (variable)
   - Use SMALL boxes (10x10 pixels) centered on intersection

### Annotation Tips:

- **Consistency:** Use same keypoints across frames
- **Visibility:** Only label keypoints that are clearly visible
- **Coverage:** Annotate at least 200-300 frames (more is better)
- **Diversity:** Include different game situations:
  - Players standing, running, jumping
  - Ball in motion, stationary, held
  - Different court positions
  - Different lighting conditions

## 2. Export Dataset from Roboflow

### Steps:

1. Go to your Roboflow project
2. Click "Export" → "YOLOv8 Format"
3. Select "Download ZIP"
4. Extract to `yolo_dataset/` directory

### Verify Export:

```bash
cd yolo_dataset/
ls -la
# Should see:
# - data.yaml
# - train/ (images + labels)
# - val/ (images + labels)
# - test/ (optional)
```

## 3. Training Dataset

```bash
# Train YOLO model
python train_yolo.py --data yolo_dataset/data.yaml --epochs 100 --batch 8

# Monitor training (if wandb enabled)
# Open: https://wandb.ai/your-workspace/yolo-training
```

## 4. Expected Performance

### Baseline (yolo11m.pt pretrained, no fine-tuning):
- Players: ~60% mAP@0.5 (generic person class)
- Ball: 0% (not in COCO dataset)
- Rim: 0% (not in COCO dataset)
- Keypoints: 0% (not in COCO dataset)

### After Fine-Tuning (target metrics):
- Players: **95%+ mAP@0.5** (basketball-specific)
- Ball: **90%+ mAP@0.5** (small object)
- Rim: **98%+ mAP@0.5** (fixed structure)
- Keypoints: **85%+ mAP@0.5** (court intersections)

## 5. Troubleshooting

### Low mAP for Ball:
- Ball is small (~10-20px diameter in 1080p)
- Solution: Increase annotations, use smaller anchor boxes
- Augmentation: Increase scale variation

### Keypoint Detection Issues:
- Keypoints are tiny (10x10px boxes)
- Solution: Increase `conf` threshold during inference
- Alternative: Use SAM2 for precise localization

### Class Imbalance:
- Players: 5-10 per frame
- Ball: 1 per frame
- Rim: 1-2 per frame
- Keypoints: 10-15 per frame
- Solution: YOLO handles this automatically with loss weighting

## 6. Example Roboflow Project Structure

```
basketball-court-tracking/
├── Version 1
│   ├── train/ (160 frames, 80%)
│   ├── valid/ (40 frames, 20%)
│   └── test/ (optional)
├── Preprocessing
│   ├── Auto-Orient: Applied
│   ├── Resize: Fit within 1920x1920
│   └── Grayscale: Off
└── Augmentations (optional)
    ├── Horizontal Flip: 50%
    ├── Brightness: ±20%
    └── Saturation: ±30%
```

## 7. Quality Checks Before Export

### Verify Annotations:
- [ ] All players labeled in every frame
- [ ] Ball labeled when visible
- [ ] Rim labeled (1-2 per frame)
- [ ] Consistent keypoints labeled (same intersections across frames)
- [ ] No missing labels (check frames with low object counts)
- [ ] Bounding boxes are tight (no excessive padding)

### Spot Check:
- Review 10% of frames randomly
- Look for:
  - Mislabeled objects (wrong class)
  - Missing objects (unlabeled players/ball)
  - Poor bounding boxes (too loose or tight)

### Export Checklist:
- [ ] Dataset split: 80/20 (train/val) or 70/15/15 (train/val/test)
- [ ] Export format: YOLOv8
- [ ] Download location: `yolo_dataset/`
- [ ] Verify `data.yaml` exists after extraction
- [ ] Verify image counts match Roboflow (train/val folders)

---

**Once exported, proceed to training:**

```bash
# Copy template config
cp yolo_dataset/data.yaml.template yolo_dataset/data.yaml

# Verify paths in data.yaml
cat yolo_dataset/data.yaml

# Start training
python train_yolo.py --data yolo_dataset/data.yaml --epochs 100 --batch 8
```
