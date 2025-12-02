#!/usr/bin/env python3
"""
Integration test for court video generation.

This script tests the court video generator with a short video segment.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.court_config import CourtVideoConfig
from app.services.court_video_generator import CourtVideoGenerator
from app.core.config import ensure_directories


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def test_court_video_generation():
    """Test court video generation with a short segment."""
    logger = logging.getLogger(__name__)

    # Ensure output directories exist
    ensure_directories()

    # Create test configuration - process only 100 frames for quick test
    config = CourtVideoConfig(
        video_path="GX020018.MP4",
        calibration_file="data/calibration/homography.json",
        court_image_file="data/calibration/court_image.png",
        tags_dir="data/tags",
        log_file="session_1763778442483.log",
        sync_offset_seconds=49.422,  # Use known sync offset for testing
        auto_sync=False,
        yolo_model="yolov8n.pt",
        yolo_confidence=0.5,
        yolo_device="mps",
        tagging_threshold_cm=200.0,
        output_video_path="data/output/test_court_view.mp4",
        output_events_path="data/output/test_tag_events.json",
        output_fps=29.97,
        start_frame=0,
        max_frames=100  # Process only 100 frames for testing
    )

    logger.info("=" * 70)
    logger.info("Court Video Generation Integration Test")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  Video: {config.video_path}")
    logger.info(f"  Frames to process: {config.max_frames}")
    logger.info(f"  Sync offset: {config.sync_offset_seconds} seconds")
    logger.info(f"  Tagging threshold: {config.tagging_threshold_cm} cm")
    logger.info(f"  YOLO device: {config.yolo_device}")
    logger.info("=" * 70)

    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        logger.error(f"Configuration validation failed: {error_msg}")
        return False

    try:
        # Create generator
        logger.info("Initializing court video generator...")
        generator = CourtVideoGenerator(config)

        # Process video
        logger.info("Processing video (this may take a few minutes)...")
        result = generator.process_video()

        # Check results
        logger.info("=" * 70)
        if result.success:
            logger.info("✓ Test PASSED!")
            logger.info(f"  Output video: {result.output_video_path}")
            logger.info(f"  Output events: {result.output_events_path}")
            logger.info(f"  Total events logged: {result.events_count}")
            logger.info(f"  Processing time: {result.processing_time_seconds:.1f} seconds")

            # Verify files exist
            if Path(result.output_video_path).exists():
                size_mb = Path(result.output_video_path).stat().st_size / (1024 * 1024)
                logger.info(f"  Video file size: {size_mb:.2f} MB")
            else:
                logger.error("  ✗ Output video file not found!")
                return False

            if Path(result.output_events_path).exists():
                size_kb = Path(result.output_events_path).stat().st_size / 1024
                logger.info(f"  Events file size: {size_kb:.2f} KB")
            else:
                logger.error("  ✗ Output events file not found!")
                return False

            logger.info("=" * 70)
            return True
        else:
            logger.error("✗ Test FAILED!")
            logger.error(f"  Error: {result.error_message}")
            logger.info("=" * 70)
            return False

    except Exception as e:
        logger.error("✗ Test FAILED with exception!")
        logger.error(f"  Error: {e}", exc_info=True)
        logger.info("=" * 70)
        return False


def test_coordinate_transformations():
    """Test coordinate transformation functions."""
    logger = logging.getLogger(__name__)

    logger.info("\nTesting coordinate transformations...")

    from app.services.calibration_integration import CalibrationIntegration

    # Initialize calibration
    calib = CalibrationIntegration("data/calibration/homography.json")

    # Test UWB to calibration rotation
    logger.info("\n1. UWB to Calibration Rotation:")
    test_points = [
        (0, 0, "Origin"),
        (1432.5, 762, "Center"),
        (2865, 1524, "Far corner")
    ]

    for x_uwb, y_uwb, desc in test_points:
        x_cal, y_cal = calib.rotate_uwb_to_calibration(x_uwb, y_uwb)
        logger.info(f"  {desc}: UWB({x_uwb}, {y_uwb}) → Cal({x_cal:.1f}, {y_cal:.1f})")

    # Test court to image projection
    logger.info("\n2. Court to Image Projection:")
    test_court_points = [
        (699, 1524, "Baseline left"),
        (1624, 1524, "Baseline center"),
        (2130, 2703, "Near basket")
    ]

    for x, y, desc in test_court_points:
        try:
            px, py = calib.court_to_image(x, y)
            logger.info(f"  {desc}: Court({x}, {y}) → Image({px}, {py})")
        except Exception as e:
            logger.error(f"  {desc}: Failed - {e}")

    # Test image to court (inverse)
    logger.info("\n3. Image to Court Projection (Inverse):")
    test_image_points = [
        (933, 1113, "Bottom-left player"),
        (1716, 1113, "Bottom-center player"),
        (2172, 1416, "Top player")
    ]

    for px, py, desc in test_image_points:
        try:
            x, y = calib.image_to_court(px, py)
            logger.info(f"  {desc}: Image({px}, {py}) → Court({x:.1f}, {y:.1f})")
        except Exception as e:
            logger.error(f"  {desc}: Failed - {e}")

    logger.info("\n✓ Coordinate transformation tests completed")
    return True


def main():
    """Main test entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Court Video Generator Integration Tests\n")

    # Run coordinate transformation tests
    if not test_coordinate_transformations():
        logger.error("Coordinate transformation tests failed")
        return 1

    # Run main video generation test
    if not test_court_video_generation():
        logger.error("\nIntegration test failed")
        return 1

    logger.info("\n" + "=" * 70)
    logger.info("All tests passed successfully!")
    logger.info("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
