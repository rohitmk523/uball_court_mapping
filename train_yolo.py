#!/usr/bin/env python3
"""
YOLO Training Script for Basketball Court Detection

Fine-tunes yolo11m.pt on custom Roboflow dataset with 4 classes:
- Class 0: players
- Class 1: ball
- Class 2: rim
- Class 3: court_keypoints

Usage:
    python train_yolo.py --data yolo_dataset/data.yaml --epochs 100 --batch 8

Features:
- Resume training from checkpoint
- Validation visualization
- Early stopping (patience=20)
- Model export to ONNX (optional)
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_yolo(
    data_yaml: str,
    base_model: str = 'yolo11m.pt',
    epochs: int = 100,
    imgsz: int = 1920,
    batch: int = 8,
    patience: int = 20,
    device: str = None,
    resume: bool = False
):
    """
    Fine-tune YOLO model on basketball court dataset.

    Args:
        data_yaml: Path to data.yaml (dataset config)
        base_model: Base YOLO model (yolo11m.pt)
        epochs: Training epochs
        imgsz: Image size (1920 for 1080p videos)
        batch: Batch size (reduce if OOM)
        patience: Early stopping patience
        device: Device ('cpu', 'cuda', 'mps', or None for auto)
        resume: Resume from last checkpoint
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            logger.info("Using Apple Silicon (MPS)")
        else:
            device = 'cpu'
            logger.info("Using CPU")

    # Load model
    if resume and Path('runs/detect/train/weights/last.pt').exists():
        logger.info("Resuming training from last checkpoint")
        model = YOLO('runs/detect/train/weights/last.pt')
    else:
        logger.info(f"Loading base model: {base_model}")
        model = YOLO(base_model)

    # Training configuration
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,

        # Optimization
        optimizer='AdamW',
        lr0=0.001,       # Initial learning rate
        lrf=0.01,        # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        hsv_h=0.015,     # HSV-Hue augmentation
        hsv_s=0.7,       # HSV-Saturation
        hsv_v=0.4,       # HSV-Value
        degrees=0.0,     # Rotation (disabled - court is fixed)
        translate=0.1,   # Translation
        scale=0.5,       # Scale
        flipud=0.0,      # Flip up-down (disabled)
        fliplr=0.5,      # Flip left-right
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.0,       # Mixup (disabled)

        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs

        # Logging
        verbose=True,
        project='runs/detect',
        name='train',
        exist_ok=True
    )

    # Save best model to checkpoints/
    best_model_path = Path('runs/detect/train/weights/best.pt')
    if best_model_path.exists():
        output_path = Path('checkpoints/yolo11m_court_finetuned.pt')
        output_path.parent.mkdir(exist_ok=True)

        import shutil
        shutil.copy(best_model_path, output_path)
        logger.info(f"Best model saved to: {output_path}")

        # Log final metrics
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        logger.info(f"Final mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        logger.info(f"Best model: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO on basketball court dataset')

    # Dataset
    parser.add_argument('--data', required=True, help='Path to data.yaml')

    # Model
    parser.add_argument('--base-model', default='yolo11m.pt', help='Base YOLO model')

    # Training params
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=1920, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    # Device
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], help='Device (auto-detect if not specified)')

    # Resume
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')

    args = parser.parse_args()

    # Verify data.yaml exists
    if not Path(args.data).exists():
        logger.error(f"data.yaml not found: {args.data}")
        logger.error("Download Roboflow dataset first:")
        logger.error("  1. Export dataset as 'YOLOv8' format from Roboflow")
        logger.error("  2. Extract to yolo_dataset/ directory")
        logger.error("  3. Verify data.yaml exists")
        return

    # Start training
    train_yolo(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
