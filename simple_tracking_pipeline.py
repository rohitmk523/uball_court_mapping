#!/usr/bin/env python3
"""
Simple Basketball Tracking Pipeline (Single-Pass)

Pipeline Flow:
1. YOLO detection (players, ball, rim, court_keypoints)
2. SAM2 segmentation (players only, always enabled)
3. ByteTrack assignment (persistent IDs)
4. Homography projection (video → court coords)
5. UWB association (proximity-based)
6. Visualization (video + court canvas)

Usage:
    python simple_tracking_pipeline.py \
        --video GX010018_1080p.MP4 \
        --start 00:26:00 \
        --end 00:27:00 \
        --yolo-model checkpoints/yolo11m_court_finetuned.pt \
        --sync-offset 1194.0 \
        --output output.mp4

Features:
- Fine-tuned YOLO for 4-class detection
- Always-on SAM2 for player segmentation
- ByteTrack for persistent IDs
- UWB association with data cleaning
- Side-by-side visualization (optional)
"""

import argparse
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Import core services
from app.services.player_detector import PlayerDetector
from app.services.calibration import get_homography_matrix, inverse_transform_point
from app.services.uwb_associator import UWBAssociator
from app.services.video_stitcher import VideoStitcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePipeline:
    """Simple single-pass basketball tracking pipeline."""

    def __init__(
        self,
        yolo_model: str,
        calibration_file: str,
        tags_dir: str,
        sync_offset: float,
        device: str = None,
        enable_stitched: bool = True,
        court_dxf: str = 'court_2.dxf',
        court_image: str = 'data/calibration/court_image.png'
    ):
        """
        Initialize pipeline components.

        Args:
            yolo_model: Path to fine-tuned YOLO model
            calibration_file: Path to homography.json
            tags_dir: Directory with UWB tag JSON files
            sync_offset: Video-to-UWB sync offset (seconds)
            device: 'cpu', 'cuda', 'mps', or None (auto-detect)
            enable_stitched: Enable side-by-side visualization
            court_dxf: Court DXF file path
            court_image: Court image for canvas
        """
        self.sync_offset = sync_offset
        self.enable_stitched = enable_stitched

        # Auto-detect device
        if device is None:
            device = self._auto_detect_device()
        self.device = device

        logger.info("=" * 60)
        logger.info("INITIALIZING PIPELINE COMPONENTS")
        logger.info("=" * 60)

        # 1. Player Detector (YOLO + SAM2 + ByteTrack)
        logger.info(f"Loading YOLO model: {yolo_model}")
        self.detector = PlayerDetector(
            model_name=yolo_model,
            confidence_threshold=0.3,
            device=device,
            enable_tracking=True,
            use_sam2=True  # Always enabled
        )

        # 2. Calibration (Homography)
        logger.info(f"Loading calibration: {calibration_file}")
        self.homography = get_homography_matrix(Path(calibration_file))
        if self.homography is None:
            raise FileNotFoundError(f"Calibration not found: {calibration_file}")

        # 3. UWB Associator
        logger.info(f"Loading UWB data: {tags_dir}")
        self.uwb_associator = UWBAssociator(
            tags_dir=Path(tags_dir),
            sync_offset_seconds=sync_offset,
            proximity_threshold=200.0,  # pixels
            canvas_width=4261,
            canvas_height=7341,
            enable_uwb_cleaning=True  # Outlier rejection + interpolation + smoothing
        )

        # 4. Video Stitcher (optional)
        self.stitcher = None
        if enable_stitched:
            logger.info("Initializing video stitcher")
            self.stitcher = VideoStitcher(
                court_image_path=court_image,
                court_dxf_path=court_dxf,
                output_height=1080,
                proximity_radius_cm=200.0
            )

        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': {'players': 0, 'ball': 0, 'rim': 0, 'keypoints': 0},
            'total_tracked': 0,
            'total_associated': 0,
            'processing_time': 0.0
        }

        logger.info("✅ All components initialized")

    def _auto_detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def process_video(
        self,
        video_path: str,
        start_time: str,
        end_time: str,
        output_path: str,
        show_preview: bool = False
    ):
        """
        Process video with tracking pipeline.

        Args:
            video_path: Input video path
            start_time: Start time (HH:MM:SS or seconds)
            end_time: End time (HH:MM:SS or seconds)
            output_path: Output video path
            show_preview: Show live preview window
        """
        # Parse time range
        start_sec = self._parse_time(start_time)
        end_sec = self._parse_time(end_time)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        total_frames = end_frame - start_frame

        logger.info("=" * 60)
        logger.info(f"Video: {video_path}")
        logger.info(f"Resolution: {width}x{height} @ {fps:.2f} FPS")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Frames: {start_frame} to {end_frame} ({total_frames} frames)")
        logger.info("=" * 60)

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Output video
        output_width = width
        if self.enable_stitched:
            court_scale = 1080 / self.stitcher.vert_h
            court_panel_w = int(self.stitcher.vert_w * court_scale)
            output_width = 1920 + court_panel_w

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))

        # Process frames
        current_frame = start_frame
        frame_count = 0
        start_time_tick = cv2.getTickCount()

        try:
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                output_frame = self._process_frame(
                    frame, current_frame, fps
                )

                # Write output
                writer.write(output_frame)

                # Show preview
                if show_preview:
                    cv2.imshow("Basketball Tracking", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = (cv2.getTickCount() - start_time_tick) / cv2.getTickFrequency()
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {progress:.1f}% | "
                        f"Frame: {current_frame}/{end_frame} | "
                        f"FPS: {avg_fps:.2f}"
                    )

                frame_count += 1
                current_frame += 1

        finally:
            cap.release()
            writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        # Calculate total time
        total_time = (cv2.getTickCount() - start_time_tick) / cv2.getTickFrequency()
        self.stats['total_frames'] = frame_count
        self.stats['processing_time'] = total_time

        # Print statistics
        self._print_statistics(output_path)

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        fps: float
    ) -> np.ndarray:
        """
        Process single frame through pipeline.

        Pipeline:
        1. YOLO detection (4 classes)
        2. SAM2 segmentation (players only)
        3. ByteTrack tracking (players only)
        4. Homography projection
        5. UWB association
        6. Visualization

        Args:
            frame: Input frame (BGR)
            frame_num: Frame number
            fps: Video FPS

        Returns:
            Output frame with visualizations
        """
        # 1. Track players (YOLO + SAM2 + ByteTrack)
        tracked_players = self.detector.track_players(frame)

        # Update statistics
        self.stats['total_tracked'] += len(tracked_players)

        # 2. Project to court coordinates
        for player in tracked_players:
            bx, by = player['bottom']
            try:
                cx, cy = inverse_transform_point((bx, by), self.homography)
                player['court_x'] = float(cx)
                player['court_y'] = float(cy)
            except:
                player['court_x'] = None
                player['court_y'] = None

        # 3. UWB association
        video_time = frame_num / fps
        self.uwb_associator.associate(video_time, tracked_players)

        # Update statistics
        self.stats['total_associated'] += sum(
            1 for p in tracked_players if p.get('uwb_tag_id') is not None
        )

        # 4. Visualize
        if self.enable_stitched and self.stitcher:
            # Get UWB positions for canvas
            uwb_time = video_time + self.sync_offset
            target_dt = self.uwb_associator.uwb_start_time + timedelta(seconds=uwb_time)
            uwb_positions = self.uwb_associator.get_uwb_world_positions_at_time(target_dt)

            # Draw detections on video
            output_frame = self.detector.draw_detections(
                frame.copy(),
                tracked_players,
                show_masks=True
            )

            # Stitch with court canvas
            output_frame = self.stitcher.create_stitched_frame(
                output_frame,
                tracked_players,
                uwb_positions,
                None  # No persistent ID mapper in single-pass
            )
        else:
            # Video-only mode
            output_frame = self.detector.draw_detections(
                frame.copy(),
                tracked_players,
                show_masks=True
            )

        return output_frame

    def _parse_time(self, time_str: str) -> float:
        """Parse time string (HH:MM:SS or seconds)."""
        if ':' in time_str:
            parts = time_str.split(':')
            hours = int(parts[0]) if len(parts) > 2 else 0
            minutes = int(parts[-2]) if len(parts) > 1 else 0
            seconds = float(parts[-1])
            return hours * 3600 + minutes * 60 + seconds
        return float(time_str)

    def _print_statistics(self, output_path: str):
        """Print processing statistics."""
        uwb_stats = self.uwb_associator.get_statistics()

        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output: {output_path}")
        logger.info(f"Total frames: {self.stats['total_frames']}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f}s")
        logger.info(f"Average FPS: {self.stats['total_frames']/self.stats['processing_time']:.2f}")
        logger.info("")
        logger.info("Detection Statistics:")
        logger.info(f"  Total players tracked: {self.stats['total_tracked']}")
        logger.info(f"  UWB associations: {self.stats['total_associated']}")
        logger.info(f"  Association rate: {uwb_stats.get('success_rate_percent', 0):.1f}%")
        logger.info("")
        logger.info("UWB Association:")
        logger.info(f"  Active mappings: {uwb_stats.get('active_mappings', 0)}")
        logger.info(f"  Track → Tag mapping: {uwb_stats.get('track_to_tag_mapping', {})}")


def main():
    parser = argparse.ArgumentParser(
        description='Simple Basketball Tracking Pipeline (Single-Pass)'
    )

    # Video
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--start', required=True, help='Start time (HH:MM:SS or seconds)')
    parser.add_argument('--end', required=True, help='End time (HH:MM:SS or seconds)')
    parser.add_argument('--output', help='Output video path (auto-generated if not specified)')

    # Model
    parser.add_argument('--yolo-model', default='checkpoints/yolo11m_court_finetuned.pt',
                        help='Fine-tuned YOLO model (default: yolo11m_court_finetuned.pt)')

    # Calibration & UWB
    parser.add_argument('--calibration', default='data/calibration/1080p/homography.json',
                        help='Homography calibration file')
    parser.add_argument('--tags-dir', default='data/tags',
                        help='UWB tags directory')
    parser.add_argument('--sync-offset', type=float, required=True,
                        help='Video-to-UWB sync offset (seconds)')

    # Visualization
    parser.add_argument('--no-stitched', action='store_true',
                        help='Disable side-by-side visualization (video only)')
    parser.add_argument('--show-preview', action='store_true',
                        help='Show live preview window')

    # Court files
    parser.add_argument('--court-dxf', default='court_2.dxf',
                        help='Court DXF file')
    parser.add_argument('--court-image', default='data/calibration/court_image.png',
                        help='Court canvas image')

    # Device
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'],
                        help='Inference device (auto-detect if not specified)')

    args = parser.parse_args()

    # Auto-generate output path
    if not args.output:
        video_name = Path(args.video).stem
        start_fmt = args.start.replace(':', '-')
        end_fmt = args.end.replace(':', '-')
        suffix = '_stitched' if not args.no_stitched else ''
        args.output = f"{video_name}_tracked_{start_fmt}_{end_fmt}{suffix}.mp4"

    # Initialize pipeline
    pipeline = SimplePipeline(
        yolo_model=args.yolo_model,
        calibration_file=args.calibration,
        tags_dir=args.tags_dir,
        sync_offset=args.sync_offset,
        device=args.device,
        enable_stitched=not args.no_stitched,
        court_dxf=args.court_dxf,
        court_image=args.court_image
    )

    # Process video
    pipeline.process_video(
        video_path=args.video,
        start_time=args.start,
        end_time=args.end,
        output_path=args.output,
        show_preview=args.show_preview
    )


if __name__ == '__main__':
    main()
