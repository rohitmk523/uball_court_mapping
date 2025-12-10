#!/usr/bin/env python3
"""
Complete Tracked Video Generator with ByteTrack + UWB Association

Generates court visualization showing:
- Green dots: HIGH confidence Track-UWB associations
- Yellow dots: NEW associations
- Red dots: LOW confidence (no UWB tag)
- Blue dots: Ghost UWB tags (no player)

Usage:
  python generate_tracked_video_complete.py \
      --video GX010018_1080p.MP4 \
      --start 00:20:00 \
      --end 00:25:00 \
      --sync-offset 8 \
      --output tracked_20_25.mp4 \
      --json-output tracked_20_25.json
"""

import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration
from app.services.uwb_associator import UWBAssociator

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Generate tracked video with persistent IDs and UWB association',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument('--video', type=str, required=True, help='Input video path')
parser.add_argument('--start', type=str, required=True, help='Start time HH:MM:SS')
parser.add_argument('--end', type=str, required=True, help='End time HH:MM:SS')
parser.add_argument('--sync-offset', type=float, required=True, help='Sync offset in seconds (video → UWB)')
parser.add_argument('--calibration', type=str, default='data/calibration/1080p/homography.json',
                   help='Calibration file path')
parser.add_argument('--tags-dir', type=str, default='data/tags', help='UWB tags directory')
parser.add_argument('--output', type=str, required=True, help='Output video filename')
parser.add_argument('--json-output', type=str, required=True, help='Output JSON log filename')

args = parser.parse_args()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_time(time_str):
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

start_seconds = parse_time(args.start)
end_seconds = parse_time(args.end)
duration_seconds = end_seconds - start_seconds

if duration_seconds <= 0:
    print("ERROR: End time must be after start time")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

COURT_IMAGE = Path("data/calibration/court_image.png")
FPS = 29.97

# Dot visualization parameters
DOT_RADIUS = 40
DOT_OUTLINE = 8

# Color coding for associations
COLOR_HIGH_CONFIDENCE = (0, 255, 0)  # Green - HIGH confidence
COLOR_NEW_ASSOCIATION = (0, 255, 255)  # Yellow - NEW association
COLOR_LOW_CONFIDENCE = (0, 0, 255)  # Red - LOW/no association
COLOR_GHOST_TAG = (255, 100, 0)  # Blue - UWB tag with no detection

print("=" * 70)
print("TRACKED VIDEO GENERATOR (ByteTrack + UWB Association)")
print("=" * 70)
print(f"Video: {args.video}")
print(f"Time window: {args.start} to {args.end}")
print(f"Duration: {duration_seconds / 60:.1f} minutes")
print(f"Sync offset: {args.sync_offset}s")
print(f"Output video: {args.output}")
print(f"Output JSON: {args.json_output}")
print("=" * 70)

# ============================================================================
# INITIALIZATION
# ============================================================================

# Load court image
print("\n1. Loading court image...")
court_canvas_orig = cv2.imread(str(COURT_IMAGE))
if court_canvas_orig is None:
    print(f"ERROR: Could not load court image from {COURT_IMAGE}")
    sys.exit(1)

# Rotate 90° clockwise (calibration was done on rotated image)
court_canvas_template = cv2.rotate(court_canvas_orig, cv2.ROTATE_90_CLOCKWISE)
CANVAS_HEIGHT, CANVAS_WIDTH = court_canvas_template.shape[:2]
print(f"   Court canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT} (rotated 90° CW)")

# Initialize player detector with tracking
print("\n2. Initializing player detector with ByteTrack tracking...")
player_detector = PlayerDetector(
    model_name="yolov8n.pt",
    confidence_threshold=0.3,
    enable_tracking=True,
    track_buffer=50
)

# Initialize calibration
print("   Loading calibration...")
calibration = CalibrationIntegration(args.calibration)

# Initialize UWB associator
print("   Initializing UWB associator...")
uwb_associator = UWBAssociator(
    tags_dir=Path(args.tags_dir),
    sync_offset_seconds=args.sync_offset,
    proximity_threshold=2.0,
    remapping_cooldown=30
)

# Open video
print(f"\n3. Opening video: {args.video}")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"ERROR: Could not open video {args.video}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"   Video: {total_frames} frames at {video_fps} fps")

# Calculate frame range
start_frame = int(start_seconds * video_fps)
end_frame = int(end_seconds * video_fps)
num_frames = end_frame - start_frame

print(f"   Processing frames {start_frame} to {end_frame} ({num_frames} frames)")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, video_fps, (CANVAS_WIDTH, CANVAS_HEIGHT))

if not out.isOpened():
    print("ERROR: Could not create video writer")
    sys.exit(1)

# JSON log structure
tracking_log = {
    "metadata": {
        "video_path": args.video,
        "calibration_file": args.calibration,
        "tags_directory": args.tags_dir,
        "sync_offset_seconds": args.sync_offset,
        "start_time": args.start,
        "end_time": args.end,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": video_fps,
        "canvas_size": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
        "generated_at": datetime.now().isoformat()
    },
    "frames": []
}

# ============================================================================
# PROCESSING LOOP
# ============================================================================

print(f"\n4. Processing {num_frames} frames with tracking and UWB association...")

for frame_idx in tqdm(range(num_frames), desc="   Processing"):
    frame_number = start_frame + frame_idx
    video_time = frame_number / video_fps

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print(f"\nEnd of video at frame {frame_number}")
        break

    # Use court image as canvas
    canvas = court_canvas_template.copy()

    # Detect and track players (with persistent IDs)
    tracked_players = player_detector.track_players(frame)

    # Project to court coordinates
    for player in tracked_players:
        bx, by = player['bottom']
        try:
            canvas_x, canvas_y = calibration.image_to_court(bx, by)
            player['court_x'] = float(canvas_x)
            player['court_y'] = float(canvas_y)
        except Exception as e:
            player['court_x'] = None
            player['court_y'] = None

    # Associate with UWB tags
    tracked_players = uwb_associator.associate(video_time, tracked_players)

    # Visualize on court canvas
    frame_data = {
        "frame_number": frame_number,
        "video_time": video_time,
        "players": []
    }

    for player in tracked_players:
        track_id = player.get('track_id')
        uwb_tag_id = player.get('uwb_tag_id')
        confidence = player.get('association_confidence', 'UNKNOWN')
        court_x = player.get('court_x')
        court_y = player.get('court_y')

        # Log player data
        player_data = {
            "track_id": track_id,
            "uwb_tag_id": uwb_tag_id,
            "bbox_video": player.get('bbox'),
            "position_court": {"x": court_x, "y": court_y} if court_x and court_y else None,
            "detection_confidence": float(player.get('confidence', 0)),
            "association_confidence": confidence
        }
        frame_data["players"].append(player_data)

        # Draw on canvas if we have court coordinates
        if court_x is not None and court_y is not None:
            canvas_x, canvas_y = int(court_x), int(court_y)

            # Check bounds
            if 0 <= canvas_x < CANVAS_WIDTH and 0 <= canvas_y < CANVAS_HEIGHT:
                # Choose color based on association confidence
                if confidence == 'HIGH':
                    color = COLOR_HIGH_CONFIDENCE
                elif confidence == 'NEW':
                    color = COLOR_NEW_ASSOCIATION
                else:
                    color = COLOR_LOW_CONFIDENCE

                # Draw dot
                cv2.circle(canvas, (canvas_x, canvas_y), DOT_RADIUS, color, -1)
                cv2.circle(canvas, (canvas_x, canvas_y), DOT_RADIUS, (255, 255, 255), DOT_OUTLINE)

                # Draw label: T{track_id}|U{tag_id} or T{track_id}|U?
                if uwb_tag_id is not None:
                    label = f"T{track_id}|U{uwb_tag_id}"
                else:
                    label = f"T{track_id}|U?"

                font_scale = 0.6
                thickness = 1
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                text_x = canvas_x - w // 2
                text_y = canvas_y - DOT_RADIUS - 10

                # Draw text shadow
                cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (0, 0, 0), thickness + 2)
                # Draw text
                cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (255, 255, 255), thickness)

    tracking_log["frames"].append(frame_data)

    # Write frame
    out.write(canvas)

# ============================================================================
# CLEANUP AND STATISTICS
# ============================================================================

cap.release()
out.release()

# Save JSON log
print(f"\n5. Saving tracking data to: {args.json_output}")
with open(args.json_output, 'w') as f:
    json.dump(tracking_log, f, indent=2)

# Print statistics
tracking_stats = player_detector.get_track_statistics()
uwb_stats = uwb_associator.get_statistics()

print("\n" + "=" * 70)
print("PROCESSING COMPLETE!")
print("=" * 70)
print(f"\nOutput files:")
print(f"  Video: {args.output}")
print(f"  Data:  {args.json_output}")

print(f"\nTracking Statistics:")
print(f"  Frames processed: {len(tracking_log['frames'])}")
print(f"  Total tracks: {tracking_stats['total_tracks']}")
print(f"  Active tracks: {tracking_stats['active_tracks']}")

print(f"\nUWB Association Statistics:")
print(f"  Active mappings: {uwb_stats['active_mappings']}")
print(f"  Success rate: {uwb_stats['success_rate_percent']:.1f}%")

if uwb_stats['track_to_tag_mapping']:
    print(f"\nTrack→Tag Mappings:")
    for track_id, tag_id in sorted(uwb_stats['track_to_tag_mapping'].items()):
        print(f"  Track {track_id} → Tag {tag_id}")

print("\n" + "=" * 70)
print("Color Legend:")
print("  🟢 Green  = HIGH confidence (validated mapping)")
print("  🟡 Yellow = NEW association (just created)")
print("  🔴 Red    = LOW confidence (no UWB tag)")
print("=" * 70)
