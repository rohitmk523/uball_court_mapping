#!/usr/bin/env python3
"""
Test script for SAM2-enhanced player tracking.

Usage:
    # Without SAM2
    python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00

    # With SAM2 (auto-detect device)
    python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2

    # With SAM2 and custom quality
    python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2 --quality large --show-masks

    # Output naming convention:
    # {video_name}_short_{start}_{end}_{quality}_bytetrack_{sam/nosam}.mp4
    # Example: GX020018_1080p_short_00:26:00_00:27:00_large_bytetrack_sam.mp4
"""

import argparse
import cv2
import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from app.services.player_detector import PlayerDetector

# Custom logging filter to suppress verbose SAM2 messages
class SAM2LogFilter(logging.Filter):
    """Filter out verbose SAM2 internal logs"""
    def filter(self, record):
        # Suppress these specific verbose messages from SAM2
        suppressed_messages = [
            'For numpy array image, we assume (HxWxC) format',
            'Computing image embeddings for the provided image',
            'Image embeddings computed'
        ]
        return not any(msg in record.getMessage() for msg in suppressed_messages)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add filter to root logger to suppress verbose SAM2 messages
logging.getLogger().addFilter(SAM2LogFilter())

# Suppress other noisy libraries
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# SAM2 model configurations
SAM2_MODELS = {
    'tiny': {
        'config': 'sam2_hiera_t.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_tiny.pt'
    },
    'small': {
        'config': 'sam2_hiera_s.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_small.pt'
    },
    'base': {
        'config': 'sam2_hiera_b+.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_base_plus.pt'
    },
    'large': {
        'config': 'sam2_hiera_l.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_large.pt'
    }
}


def auto_detect_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"🚀 Auto-detected NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logger.info("🚀 Auto-detected Apple Silicon (MPS)")
    else:
        device = 'cpu'
        logger.info("💻 Using CPU (no GPU detected)")
    return device


def parse_time(time_str):
    """Parse time string (HH:MM:SS or seconds) to seconds"""
    if ':' in time_str:
        parts = time_str.split(':')
        hours = int(parts[0]) if len(parts) > 2 else 0
        minutes = int(parts[-2]) if len(parts) > 1 else 0
        seconds = float(parts[-1])
        return hours * 3600 + minutes + seconds
    return float(time_str)


def format_time_for_filename(time_str):
    """Format time string for filename (replace : with -)"""
    return time_str.replace(':', '-')


def generate_output_filename(video_path, start_time, end_time, quality, use_sam2):
    """
    Generate output filename based on convention:
    {video_name}_short_{start}_{end}_{quality}_bytetrack_{sam/nosam}.mp4

    Args:
        video_path: Path to input video
        start_time: Start time string (HH:MM:SS)
        end_time: End time string (HH:MM:SS)
        quality: SAM model quality (tiny, small, base, large)
        use_sam2: Whether SAM2 is enabled

    Returns:
        Generated filename string
    """
    video_name = Path(video_path).stem  # Get filename without extension
    start_fmt = format_time_for_filename(start_time)
    end_fmt = format_time_for_filename(end_time)
    segmentation = 'sam' if use_sam2 else 'nosam'

    filename = f"{video_name}_short_{start_fmt}_{end_fmt}_{quality}_bytetrack_{segmentation}.mp4"
    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Test SAM2-enhanced player tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Without SAM2
  python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00

  # With SAM2 (default quality: large)
  python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2

  # With SAM2 and show masks overlay
  python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2 --show-masks

  # Custom quality model
  python test_sam2_tracking.py --video GX010018_1080p.MP4 --start 00:26:00 --end 00:27:00 --sam2 --quality base

Note:
  --show-masks: When enabled, displays segmentation masks as colored overlays instead of bounding boxes.
                Provides precise player outlines for better visualization. Only works with --sam2.
        """
    )
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--start', required=True, help='Start time (HH:MM:SS or seconds)')
    parser.add_argument('--end', required=True, help='End time (HH:MM:SS or seconds)')
    parser.add_argument('--output', help='Output video path (auto-generated if not specified)')
    parser.add_argument('--sam2', action='store_true', help='Enable SAM2 segmentation')
    parser.add_argument('--quality', default='large', choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model quality (default: large). Higher quality = better but slower.')
    parser.add_argument('--device', default=None, choices=['cpu', 'cuda', 'mps'],
                        help='Device for inference (default: auto-detect)')
    parser.add_argument('--yolo-model', default='yolo11n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--show-masks', action='store_true',
                        help='Show segmentation masks as colored overlays (requires --sam2)')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')

    args = parser.parse_args()

    # Validate show-masks requires sam2
    if args.show_masks and not args.sam2:
        logger.warning("--show-masks requires --sam2. Ignoring --show-masks flag.")
        args.show_masks = False

    # Auto-detect device if not specified
    if args.device is None:
        args.device = auto_detect_device()
    else:
        logger.info(f"Using manually specified device: {args.device}")

    # Parse start and end times
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)

    # Calculate duration
    duration_seconds = end_seconds - start_seconds
    if duration_seconds <= 0:
        logger.error(f"End time ({args.end}) must be after start time ({args.start})")
        return

    logger.info(f"Time range: {args.start} to {args.end} (duration: {duration_seconds:.1f}s)")

    # Update SAM2 config if using SAM2
    sam2_config_path = None
    if args.sam2:
        # Update config file with selected quality
        config_path = Path('config/sam2_config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                sam2_config = json.load(f)

            # Update model configuration based on quality
            model_info = SAM2_MODELS[args.quality]
            sam2_config['sam2']['model_cfg'] = model_info['config']
            sam2_config['sam2']['checkpoint_path'] = model_info['checkpoint']
            sam2_config['sam2']['device'] = args.device

            # Write updated config to temp file
            temp_config_path = Path('config/sam2_config_temp.json')
            with open(temp_config_path, 'w') as f:
                json.dump(sam2_config, f, indent=2)

            sam2_config_path = str(temp_config_path)
            logger.info(f"Using SAM2 quality: {args.quality} ({model_info['config']})")
        else:
            logger.warning(f"SAM2 config not found: {config_path}")

    # Initialize video capture
    logger.info(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Seek to start time
    start_frame = int(start_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    logger.info(f"Starting at frame {start_frame} ({start_seconds:.1f}s)")

    # Calculate end frame
    duration_frames = int(duration_seconds * fps)
    end_frame = start_frame + duration_frames
    logger.info(f"Processing {duration_frames} frames ({duration_seconds:.1f}s)")

    # Initialize player detector
    logger.info("Initializing player detector...")
    logger.info(f"SAM2 enabled: {args.sam2}")
    detector = PlayerDetector(
        model_name=args.yolo_model,
        confidence_threshold=args.confidence,
        device=args.device,
        enable_tracking=True,
        use_sam2=args.sam2,
        sam2_config_path=sam2_config_path
    )

    # Auto-generate output filename if not specified
    if not args.output:
        args.output = generate_output_filename(
            args.video,
            args.start,
            args.end,
            args.quality,
            args.sam2
        )
        logger.info(f"Auto-generated output filename: {args.output}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    logger.info(f"Output video: {args.output}")

    # Processing statistics
    frame_count = 0
    total_time = 0.0
    current_frame = start_frame

    logger.info("=" * 60)
    logger.info("Processing video...")
    logger.info("=" * 60)

    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, stopping")
                break

            # Process frame
            start_time = cv2.getTickCount()

            # Track players with SAM2 (if enabled)
            tracked_players = detector.track_players(frame)

            # Calculate processing time
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            total_time += elapsed_time

            # Visualize
            output_frame = detector.draw_detections(
                frame.copy(),
                tracked_players,
                show_masks=args.show_masks
            )

            # Draw info overlay
            info_text = [
                f"Frame: {current_frame}/{end_frame}",
                f"Players: {len(tracked_players)}",
                f"FPS: {1.0/elapsed_time:.1f}",
                f"SAM2: {'ON' if detector.use_sam2 else 'OFF'}"
            ]

            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(
                    output_frame,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # Write to output video
            if writer:
                writer.write(output_frame)

            # Display frame
            if not args.no_display:
                cv2.imshow("SAM2 Tracking Test", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit")
                    break

            # Update counters
            frame_count += 1
            current_frame += 1

            # Log progress
            if frame_count % 30 == 0:
                progress = (frame_count / duration_frames) * 100
                avg_fps = frame_count / total_time
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Frames: {frame_count}/{duration_frames} | "
                    f"Avg FPS: {avg_fps:.2f}"
                )

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Print final statistics
    logger.info("=" * 60)
    logger.info("Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average FPS: {frame_count/total_time:.2f}")
    logger.info(f"SAM2 enabled: {detector.use_sam2}")

    if detector.tracker:
        stats = detector.get_track_statistics()
        logger.info(f"Total tracks: {stats['total_tracks']}")
        logger.info(f"Active tracks: {stats['active_tracks']}")

    if args.output:
        logger.info(f"Output saved: {args.output}")


if __name__ == "__main__":
    main()
