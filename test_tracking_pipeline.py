#!/usr/bin/env python3
"""
Quick tracking test script - Verify ByteTrack + UWB association works
Usage: python test_tracking_pipeline.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:21:00 --sync-offset 8
"""

import sys
import cv2
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.player_detector import PlayerDetector
from app.services.uwb_associator import UWBAssociator
from app.services.calibration_integration import CalibrationIntegration

def parse_time(time_str):
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

def main():
    parser = argparse.ArgumentParser(description='Test ByteTrack tracking with UWB association')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--start', required=True, help='Start time HH:MM:SS')
    parser.add_argument('--end', required=True, help='End time HH:MM:SS')
    parser.add_argument('--sync-offset', type=float, required=True, help='Sync offset in seconds')
    args = parser.parse_args()

    print("="*70)
    print("TRACKING TEST - ByteTrack + UWB Association")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Time: {args.start} to {args.end}")
    print(f"Sync offset: {args.sync_offset}s")
    print("="*70)

    # Initialize components
    print("\n1. Initializing tracking system...")
    detector = PlayerDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.3,
        enable_tracking=True,
        track_buffer=50
    )
    print("   ✓ PlayerDetector initialized with ByteTrack")

    calibration = CalibrationIntegration("data/calibration/1080p/homography.json")
    print("   ✓ Calibration loaded")

    uwb_associator = UWBAssociator(
        tags_dir=Path("data/tags"),
        sync_offset_seconds=args.sync_offset,
        proximity_threshold=2.0
    )
    print("   ✓ UWB Associator initialized")

    # Open video
    print("\n2. Opening video...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   ✓ Video opened: {fps:.2f} fps")

    # Parse times and set position
    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)
    duration_sec = end_sec - start_sec

    start_frame = int(start_sec * fps)
    max_frames = int(duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"   Starting at frame {start_frame}")
    print(f"   Processing {max_frames} frames ({duration_sec}s)")

    frame_count = 0
    print("\n3. Processing video frames...")
    print("   (Showing updates every 30 frames / ~1 second)\n")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"\n   End of video at frame {frame_count}")
            break

        # Track players with persistent IDs
        tracked_players = detector.track_players(frame)

        # Project to court coordinates
        for player in tracked_players:
            bx, by = player['bottom']
            try:
                court_x, court_y = calibration.image_to_court(bx, by)
                player['court_x'] = float(court_x)
                player['court_y'] = float(court_y)
            except:
                player['court_x'] = None
                player['court_y'] = None

        # Associate with UWB tags
        video_time = (start_frame + frame_count) / fps
        tracked_players = uwb_associator.associate(video_time, tracked_players)

        # Print tracking info every second
        if frame_count % 30 == 0:
            current_time = start_sec + (frame_count / fps)
            time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            print(f"   Frame {frame_count} (Time: {time_str}) - {len(tracked_players)} players:")

            for p in tracked_players:
                tid = p.get('track_id')
                uid = p.get('uwb_tag_id')
                conf = p.get('association_confidence', 'N/A')
                court_x = p.get('court_x')
                court_y = p.get('court_y')

                if uid is not None:
                    if court_x and court_y:
                        print(f"      Track {tid} → UWB {uid} ({conf}) at ({court_x:.1f}, {court_y:.1f})")
                    else:
                        print(f"      Track {tid} → UWB {uid} ({conf}) [no court coords]")
                else:
                    print(f"      Track {tid} → No UWB ({conf})")

        frame_count += 1

    cap.release()

    # Print final statistics
    print("\n" + "="*70)
    print("TRACKING STATISTICS")
    print("="*70)

    track_stats = detector.get_track_statistics()
    print(f"\nByteTrack:")
    print(f"  Total tracks created: {track_stats['total_tracks']}")
    print(f"  Currently active: {track_stats['active_tracks']}")
    print(f"  Frames processed: {track_stats['frame_count']}")

    uwb_stats = uwb_associator.get_statistics()
    print(f"\nUWB Association:")
    print(f"  Association attempts: {uwb_stats['total_associations_attempted']}")
    print(f"  Successful: {uwb_stats['successful_associations']}")
    print(f"  Success rate: {uwb_stats['success_rate_percent']:.1f}%")
    print(f"  Active mappings: {uwb_stats['active_mappings']}")

    if uwb_stats['track_to_tag_mapping']:
        print(f"\nTrack → Tag Mappings:")
        for track_id, tag_id in sorted(uwb_stats['track_to_tag_mapping'].items()):
            age = uwb_stats['mapping_ages'].get(track_id, 0)
            print(f"  Track {track_id} → Tag {tag_id} (age: {age} frames)")

    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the console output above")
    print("2. Check if Track IDs are persistent across frames")
    print("3. Verify UWB associations look correct")
    print("4. If good, run full pipeline with visual output")
    print("="*70)

if __name__ == "__main__":
    main()
