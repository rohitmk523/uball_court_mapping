#!/usr/bin/env python3
"""
Standalone script to process video with court-centric visualization.

This script generates a video showing the basketball court as the primary canvas,
with players projected from video onto the court and UWB tags overlaid.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.court_config import CourtVideoConfig
from app.services.court_video_generator import CourtVideoGenerator
from app.core.config import ensure_directories


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate court-centric video with player-tag visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with auto-sync
  python process_court_video.py GX020018.MP4

  # Process with manual sync offset
  python process_court_video.py GX020018.MP4 --sync-offset 49.422

  # Process first 300 frames only
  python process_court_video.py GX020018.MP4 --max-frames 300

  # Process with custom threshold
  python process_court_video.py GX020018.MP4 --threshold 150
        """
    )

    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )

    parser.add_argument(
        '--calibration',
        type=str,
        default='data/calibration/homography.json',
        help='Path to calibration file (default: data/calibration/homography.json)'
    )

    parser.add_argument(
        '--court-image',
        type=str,
        default='data/calibration/court_image.png',
        help='Path to court image (default: data/calibration/court_image.png)'
    )

    parser.add_argument(
        '--tags-dir',
        type=str,
        default='data/tags',
        help='Directory containing tag JSON files (default: data/tags)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default='data/logs/session_1763778442483.log',
        help='UWB log file for sync (default: data/logs/session_1763778442483.log)'
    )

    parser.add_argument(
        '--sync-offset',
        type=float,
        default=None,
        help='Manual sync offset in seconds (default: auto-calculate)'
    )

    parser.add_argument(
        '--no-auto-sync',
        action='store_true',
        help='Disable auto-sync calculation'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=200.0,
        help='Tagging threshold in cm (default: 200.0)'
    )

    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model to use (default: yolov8n.pt)'
    )

    parser.add_argument(
        '--yolo-confidence',
        type=float,
        default=0.5,
        help='YOLO confidence threshold (default: 0.5)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for YOLO inference (default: mps)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/output/court_view_{timestamp}.mp4',
        help='Output video path (default: data/output/court_view_{timestamp}.mp4)'
    )

    parser.add_argument(
        '--events-output',
        type=str,
        default='data/output/tag_events_{timestamp}.json',
        help='Output events JSON path (default: data/output/tag_events_{timestamp}.json)'
    )

    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Start frame number (default: 0)'
    )

    parser.add_argument(
        '--end-frame',
        type=int,
        default=None,
        help='End frame number (default: process until end)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (default: no limit)'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=29.97,
        help='Output video FPS (default: 29.97)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Ensure output directories exist
    ensure_directories()

    # Create configuration
    config = CourtVideoConfig(
        video_path=args.video_path,
        calibration_file=args.calibration,
        court_image_file=args.court_image,
        tags_dir=args.tags_dir,
        log_file=args.log_file,
        sync_offset_seconds=args.sync_offset,
        auto_sync=not args.no_auto_sync,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_device=args.device,
        tagging_threshold_cm=args.threshold,
        output_video_path=args.output,
        output_events_path=args.events_output,
        output_fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames
    )

    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        logger.error(f"Configuration error: {error_msg}")
        return 1

    # Log configuration
    logger.info("=" * 60)
    logger.info("Court Video Processing Configuration")
    logger.info("=" * 60)
    logger.info(f"Video: {config.video_path}")
    logger.info(f"Calibration: {config.calibration_file}")
    logger.info(f"Court image: {config.court_image_file}")
    logger.info(f"Tags directory: {config.tags_dir}")
    logger.info(f"Sync offset: {'auto' if config.auto_sync else config.sync_offset} seconds")
    logger.info(f"Tagging threshold: {config.tagging_threshold_cm} cm")
    logger.info(f"YOLO model: {config.yolo_model} (confidence: {config.yolo_confidence})")
    logger.info(f"Device: {config.yolo_device}")
    logger.info(f"Output video: {config.output_video_path}")
    logger.info(f"Output events: {config.output_events_path}")
    logger.info("=" * 60)

    # Create generator
    try:
        generator = CourtVideoGenerator(config)
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}", exc_info=True)
        return 1

    # Process video
    logger.info("Starting video processing...")
    result = generator.process_video()

    # Report results
    logger.info("=" * 60)
    if result.success:
        logger.info("Processing completed successfully!")
        logger.info(f"Output video: {result.output_video_path}")
        logger.info(f"Output events: {result.output_events_path}")
        logger.info(f"Total events logged: {result.events_count}")
        logger.info(f"Processing time: {result.processing_time_seconds:.1f} seconds")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("Processing failed!")
        logger.error(f"Error: {result.error_message}")
        logger.info("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
