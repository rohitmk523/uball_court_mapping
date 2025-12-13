#!/usr/bin/env python3
"""
Persistent ID Tracking Pipeline - Two-Pass Processing System

Transforms test_sam2_tracking.py into a production-ready system with:
- Pass 1: Data collection and UWB association
- Pass 2: Video reprocessing with persistent IDs (colors tied to UWB tags)

Features:
- Dual output modes: Stitched (video + court canvas) OR video-only
- Association logging (JSON) with ID transition events
- Statistics reporting
- Color persistence across frames (tied to UWB tag_id, not ByteTrack ID)

Usage Examples:
  # Full pipeline (both passes)
  python persistent_id_tracking.py \
    --video GX010018_1080p.MP4 \
    --start 00:26:00 \
    --end 00:27:00 \
    --sync-offset 1194.0 \
    --mode both \
    --sam2 \
    --show-masks \
    --show-stitched

  # Pass 1 only (data collection)
  python persistent_id_tracking.py \
    --video GX010018_1080p.MP4 \
    --start 00:26:00 \
    --end 00:27:00 \
    --sync-offset 1194.0 \
    --mode pass1 \
    --association-log session_001.json

  # Pass 2 only (reprocessing)
  python persistent_id_tracking.py \
    --video GX010018_1080p.MP4 \
    --start 00:26:00 \
    --end 00:27:00 \
    --sync-offset 1194.0 \
    --mode pass2 \
    --load-associations session_001.json \
    --output output_persistent.mp4
"""

import argparse
import cv2
import torch
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Import project services
from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration
from app.services.uwb_associator import UWBAssociator
from app.services.association_logger import AssociationLogger
from app.services.persistent_id_mapper import PersistentIDMapper
from app.services.video_stitcher import VideoStitcher

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
logging.getLogger('ezdxf').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

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
    """Auto-detect the best available device."""
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
    """Parse time string (HH:MM:SS or seconds) to seconds."""
    if ':' in time_str:
        parts = time_str.split(':')
        hours = int(parts[0]) if len(parts) > 2 else 0
        minutes = int(parts[-2]) if len(parts) > 1 else 0
        seconds = float(parts[-1])
        return hours * 3600 + minutes * 60 + seconds
    return float(time_str)


def format_time_for_filename(time_str):
    """Format time string for filename (replace : with -)."""
    return time_str.replace(':', '-')


def generate_output_filename(
    video_path: str,
    start_time: str,
    end_time: str,
    quality: str,
    use_sam2: bool,
    stitched: bool = False
):
    """
    Generate output filename.

    Format: {video_name}_persistent_{start}_{end}_{quality}_bytetrack_{sam/nosam}[_stitched].mp4
    """
    video_name = Path(video_path).stem
    start_fmt = format_time_for_filename(start_time)
    end_fmt = format_time_for_filename(end_time)
    segmentation = 'sam' if use_sam2 else 'nosam'
    stitch_suffix = '_stitched' if stitched else ''

    filename = f"{video_name}_persistent_{start_fmt}_{end_fmt}_{quality}_bytetrack_{segmentation}{stitch_suffix}.mp4"
    return filename


def setup_sam2_config(quality: str, device: str):
    """Setup SAM2 configuration file."""
    config_path = Path('config/sam2_config.json')
    if not config_path.exists():
        logger.warning(f"SAM2 config not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        sam2_config = json.load(f)

    # Update model configuration
    model_info = SAM2_MODELS[quality]
    sam2_config['sam2']['model_cfg'] = model_info['config']
    sam2_config['sam2']['checkpoint_path'] = model_info['checkpoint']
    sam2_config['sam2']['device'] = device

    # Write updated config to temp file
    temp_config_path = Path('config/sam2_config_temp.json')
    with open(temp_config_path, 'w') as f:
        json.dump(sam2_config, f, indent=2)

    logger.info(f"✅ SAM2 config: {quality} ({model_info['config']})")
    return str(temp_config_path)


def run_pass1(args):
    """
    Pass 1: Data Collection & Association

    Pipeline:
    1. Track players with YOLO + ByteTrack + SAM2
    2. Project to court coordinates (homography)
    3. Associate with UWB tags (proximity-based)
    4. Log frame-by-frame associations

    Output:
    - association_log.json (frame-level mapping + statistics)
    """
    logger.info("=" * 60)
    logger.info("PASS 1: DATA COLLECTION & ASSOCIATION")
    logger.info("=" * 60)

    # Parse time range
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)
    duration_seconds = end_seconds - start_seconds

    logger.info(f"Time range: {args.start} to {args.end} (duration: {duration_seconds:.1f}s)")

    # Open video
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

    logger.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Calculate frame range
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    duration_frames = end_frame - start_frame

    logger.info(f"Processing frames {start_frame} to {end_frame} ({duration_frames} frames)")

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize components
    logger.info("Initializing components...")

    # Player detector (YOLO + ByteTrack + SAM2)
    detector = PlayerDetector(
        model_name=args.yolo_model,
        confidence_threshold=args.confidence,
        device=args.device,
        enable_tracking=True,
        use_sam2=args.sam2,
        sam2_config_path=args.sam2_config_path if hasattr(args, 'sam2_config_path') else None
    )

    # Calibration (homography projection)
    calibration = CalibrationIntegration(args.calibration)

    # UWB Associator
    uwb_associator = UWBAssociator(
        tags_dir=Path(args.tags_dir),
        sync_offset_seconds=args.sync_offset,
        proximity_threshold=200.0,  # pixels
        canvas_width=4261,
        canvas_height=7341
    )

    # Association Logger
    association_logger = AssociationLogger(
        video_path=args.video,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        sync_offset=args.sync_offset,
        proximity_threshold=200.0
    )

    logger.info("✅ All components initialized")

    # Process frames
    frame_count = 0
    current_frame = start_frame
    total_time = 0.0

    logger.info("=" * 60)
    logger.info("Processing frames...")
    logger.info("=" * 60)

    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, stopping")
                break

            # Start timing
            start_time = cv2.getTickCount()

            # Track players
            tracked_players = detector.track_players(frame)

            # Project to court coordinates
            for player in tracked_players:
                bx, by = player['bottom']
                try:
                    cx, cy = calibration.image_to_court(bx, by)
                    player['court_x'] = float(cx)
                    player['court_y'] = float(cy)
                except Exception as e:
                    player['court_x'] = None
                    player['court_y'] = None

            # Associate with UWB
            video_time_sec = current_frame / fps
            uwb_associator.associate(video_time_sec, tracked_players)

            # Get UWB positions for logging
            uwb_time_sec = video_time_sec + args.sync_offset
            target_dt = uwb_associator.uwb_start_time + timedelta(seconds=uwb_time_sec)
            uwb_positions = uwb_associator._get_uwb_positions_at_time(target_dt)

            # Log associations
            association_logger.log_frame(
                frame_num=current_frame,
                video_time=video_time_sec,
                uwb_time=uwb_time_sec,
                tracked_players=tracked_players,
                uwb_positions=uwb_positions
            )

            # Calculate processing time
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            total_time += elapsed_time

            # Progress logging
            if frame_count % 30 == 0:
                progress = (frame_count / duration_frames) * 100
                avg_fps = frame_count / total_time if total_time > 0 else 0
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Frame: {current_frame}/{end_frame} | "
                    f"Avg FPS: {avg_fps:.2f}"
                )

            frame_count += 1
            current_frame += 1

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        cap.release()

    # Save association log
    association_logger.save(args.association_log)

    # Print statistics
    stats = uwb_associator.get_statistics()
    logger.info("=" * 60)
    logger.info("PASS 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    if total_time > 0:
        logger.info(f"Average FPS: {frame_count/total_time:.2f}")
    logger.info(f"Association log: {args.association_log}")
    logger.info(f"UWB association rate: {stats.get('success_rate_percent', 0):.1f}%")
    logger.info(f"Active mappings: {stats.get('active_mappings', 0)}")


def run_pass2(args):
    """
    Pass 2: Video Reprocessing with Persistent IDs

    Pipeline:
    1. Load association log from Pass 1
    2. Re-run same tracking pipeline (YOLO + ByteTrack + SAM2)
    3. Apply persistent IDs from mapper (ByteTrack → UWB tag)
    4. Render with consistent colors (tied to UWB tag_id)
    5. Optional: Stitch with court canvas

    Output:
    - Reprocessed video with persistent colors and dual ID labels
    """
    logger.info("=" * 60)
    logger.info("PASS 2: VIDEO REPROCESSING WITH PERSISTENT IDS")
    logger.info("=" * 60)

    # Load association log
    assoc_file = args.load_associations or args.association_log
    if not Path(assoc_file).exists():
        logger.error(f"❌ Association log not found: {assoc_file}")
        logger.error("Run Pass 1 first, or specify --load-associations")
        return

    logger.info(f"Loading association log: {assoc_file}")
    id_mapper = PersistentIDMapper(assoc_file)

    # Parse time range
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)
    duration_seconds = end_seconds - start_seconds

    # Open video
    logger.info(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame range
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    duration_frames = end_frame - start_frame

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize components (same as Pass 1)
    logger.info("Initializing components...")

    detector = PlayerDetector(
        model_name=args.yolo_model,
        confidence_threshold=args.confidence,
        device=args.device,
        enable_tracking=True,
        use_sam2=args.sam2,
        sam2_config_path=args.sam2_config_path if hasattr(args, 'sam2_config_path') else None
    )

    calibration = CalibrationIntegration(args.calibration)

    # Initialize video stitcher (if stitched mode)
    stitcher = None
    output_width = width
    if args.show_stitched:
        stitcher = VideoStitcher(
            court_image_path=args.court_image,
            court_dxf_path=args.court_dxf,
            output_height=1080,
            proximity_radius_cm=200.0
        )
        # Calculate stitched width
        court_scale = 1080 / stitcher.vert_h
        court_panel_w = int(stitcher.vert_w * court_scale)
        output_width = 1920 + court_panel_w

    # Initialize UWB associator (for fetching UWB positions in stitched mode)
    uwb_associator = None
    if args.show_stitched:
        uwb_associator = UWBAssociator(
            tags_dir=Path(args.tags_dir),
            sync_offset_seconds=args.sync_offset,
            proximity_threshold=200.0,
            canvas_width=4261,
            canvas_height=7341
        )

    logger.info("✅ All components initialized")

    # Output video
    if not args.output:
        args.output = generate_output_filename(
            args.video,
            args.start,
            args.end,
            args.quality,
            args.sam2,
            args.show_stitched
        )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (output_width, height))

    logger.info(f"Output: {args.output}")
    logger.info(f"Stitched mode: {args.show_stitched}")
    logger.info(f"Output size: {output_width}x{height}")

    # Process frames
    frame_count = 0
    current_frame = start_frame
    total_time = 0.0

    logger.info("=" * 60)
    logger.info("Processing frames...")
    logger.info("=" * 60)

    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, stopping")
                break

            # Start timing
            start_time = cv2.getTickCount()

            # Track players (same pipeline as Pass 1)
            tracked_players = detector.track_players(frame)

            # Project to court coordinates
            for player in tracked_players:
                bx, by = player['bottom']
                try:
                    cx, cy = calibration.image_to_court(bx, by)
                    player['court_x'] = float(cx)
                    player['court_y'] = float(cy)
                except:
                    player['court_x'] = None
                    player['court_y'] = None

            # Apply persistent IDs from mapper
            for player in tracked_players:
                track_id = player['track_id']
                tag_id = id_mapper.get_persistent_id(current_frame, track_id)
                player['uwb_tag_id'] = tag_id

            # Render frame
            if args.show_stitched and stitcher and uwb_associator:
                # Get UWB positions for canvas rendering (world coordinates in cm)
                video_time_sec = current_frame / fps
                uwb_time_sec = video_time_sec + args.sync_offset
                target_dt = uwb_associator.uwb_start_time + timedelta(seconds=uwb_time_sec)
                uwb_positions = uwb_associator.get_uwb_world_positions_at_time(target_dt)

                # Draw detections on video frame
                output_frame = detector.draw_detections(
                    frame.copy(),
                    tracked_players,
                    show_masks=args.show_masks,
                    color_by_tag_id=True,
                    id_mapper=id_mapper
                )

                # Stitch with court canvas
                output_frame = stitcher.create_stitched_frame(
                    output_frame,
                    tracked_players,
                    uwb_positions,
                    id_mapper
                )
            else:
                # Video-only mode: just draw detections
                output_frame = detector.draw_detections(
                    frame.copy(),
                    tracked_players,
                    show_masks=args.show_masks,
                    color_by_tag_id=True,
                    id_mapper=id_mapper
                )

            # Write frame
            writer.write(output_frame)

            # Display (optional)
            if not args.no_display:
                cv2.imshow("Persistent ID Tracking", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break

            # Calculate processing time
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            total_time += elapsed_time

            # Progress logging
            if frame_count % 30 == 0:
                progress = (frame_count / duration_frames) * 100
                avg_fps = frame_count / total_time if total_time > 0 else 0
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Frame: {current_frame}/{end_frame} | "
                    f"Avg FPS: {avg_fps:.2f}"
                )

            frame_count += 1
            current_frame += 1

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # Print statistics
    stats = id_mapper.get_statistics()
    logger.info("=" * 60)
    logger.info("PASS 2 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    if total_time > 0:
        logger.info(f"Average FPS: {frame_count/total_time:.2f}")
    logger.info(f"Output video: {args.output}")
    logger.info(f"Unique UWB tags: {stats['unique_uwb_tags']}")
    logger.info(f"Success rate: {stats['success_rate']:.1f}%")

    # Generate statistics report
    report_path = args.output.replace('.mp4', '_statistics.md')
    video_info = {
        'video_path': args.video,
        'start_time': args.start,
        'end_time': args.end,
        'duration': duration_seconds,
        'total_frames': frame_count,
        'fps': fps
    }
    id_mapper.generate_statistics_report(report_path, video_info)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Persistent ID Tracking Pipeline (Two-Pass Processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (both passes)
  python persistent_id_tracking.py \\
    --video GX010018_1080p.MP4 \\
    --start 00:26:00 \\
    --end 00:27:00 \\
    --sync-offset 1194.0 \\
    --mode both \\
    --sam2 \\
    --show-masks \\
    --show-stitched

  # Pass 1 only (data collection)
  python persistent_id_tracking.py \\
    --video GX010018_1080p.MP4 \\
    --start 00:26:00 \\
    --end 00:27:00 \\
    --sync-offset 1194.0 \\
    --mode pass1 \\
    --association-log session_001.json

  # Pass 2 only (reprocessing)
  python persistent_id_tracking.py \\
    --video GX010018_1080p.MP4 \\
    --start 00:26:00 \\
    --end 00:27:00 \\
    --sync-offset 1194.0 \\
    --mode pass2 \\
    --load-associations session_001.json \\
    --output output_persistent.mp4
        """
    )

    # Video input
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--start', required=True, help='Start time (HH:MM:SS or seconds)')
    parser.add_argument('--end', required=True, help='End time (HH:MM:SS or seconds)')

    # UWB synchronization
    parser.add_argument('--sync-offset', type=float, required=True,
                        help='Sync offset (seconds): video_time + offset = uwb_time')
    parser.add_argument('--tags-dir', default='data/tags',
                        help='UWB tags directory (default: data/tags)')

    # Calibration
    parser.add_argument('--calibration', default='data/calibration/1080p/homography.json',
                        help='Homography calibration file')
    parser.add_argument('--court-dxf', default='court_2.dxf',
                        help='Court DXF file (for geometry)')
    parser.add_argument('--court-image', default='data/calibration/court_image.png',
                        help='Court canvas image')

    # Processing mode
    parser.add_argument('--mode', choices=['pass1', 'pass2', 'both'], default='both',
                        help='Pass 1 (collect), Pass 2 (render), or Both (default: both)')

    # Pass 1 output
    parser.add_argument('--association-log', default='association_log.json',
                        help='Association log output path (Pass 1)')

    # Pass 2 input/output
    parser.add_argument('--load-associations',
                        help='Load associations from Pass 1 (for Pass 2 only)')
    parser.add_argument('--output',
                        help='Output video path (Pass 2, auto-generated if not specified)')

    # Visualization
    parser.add_argument('--show-stitched', action='store_true',
                        help='Enable side-by-side video (video + court canvas)')

    # SAM2 options
    parser.add_argument('--sam2', action='store_true',
                        help='Enable SAM2 segmentation')
    parser.add_argument('--quality', default='large',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model quality (default: large)')
    parser.add_argument('--show-masks', action='store_true',
                        help='Show SAM2 masks as colored overlays')

    # YOLO options
    parser.add_argument('--yolo-model', default='yolo11m.pt',
                        help='YOLO model path (default: yolo11m.pt - medium model for better detection)')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Detection confidence threshold (default: 0.3)')

    # Other
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live video preview')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'],
                        help='Inference device (default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = auto_detect_device()

    # Setup SAM2 config (if enabled)
    if args.sam2:
        args.sam2_config_path = setup_sam2_config(args.quality, args.device)

    # Run requested mode
    if args.mode in ['pass1', 'both']:
        run_pass1(args)

    if args.mode in ['pass2', 'both']:
        run_pass2(args)

    logger.info("🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
