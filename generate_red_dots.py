#!/usr/bin/env python3
"""
Red dots video generator: Player detections only - No UWB tags.
Logs all player coordinates to JSON for later use.
Usage: python generate_red_dots.py --start 00:25:00 --end 00:26:00 --resolution compact
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

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description='Generate red dots (player detections only) court video with coordinate logging',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  # 1 minute from 25-26 min, compact resolution
  python generate_red_dots.py --start 00:25:00 --end 00:26:00 --resolution compact

  # 5 minutes from 30-35 min, high resolution
  python generate_red_dots.py --start 00:30:00 --end 00:35:00 --resolution highres

  # Full game duration
  python generate_red_dots.py --start 00:00:00 --end 01:00:00
'''
)

parser.add_argument(
    '--start',
    type=str,
    required=True,
    help='Start time in HH:MM:SS format (from video start)'
)

parser.add_argument(
    '--end',
    type=str,
    required=True,
    help='End time in HH:MM:SS format (from video start)'
)

parser.add_argument(
    '--resolution',
    type=str,
    choices=['compact', 'highres'],
    default='compact',
    help='Canvas resolution: compact (800x1200) or highres (4260x7340)'
)

parser.add_argument(
    '--video',
    type=str,
    default='GX020018_1080p.MP4',
    help='Input video path (default: GX020018_1080p.MP4)'
)

parser.add_argument(
    '--calibration',
    type=str,
    default='data/calibration/1080p/homography.json',
    help='Calibration file path'
)

parser.add_argument(
    '--output',
    type=str,
    default=None,
    help='Output video filename (default: auto-generated)'
)

parser.add_argument(
    '--coords-log',
    type=str,
    default=None,
    help='Output coordinates JSON filename (default: auto-generated)'
)

args = parser.parse_args()

# Parse time strings
def parse_time(time_str):
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

start_seconds = parse_time(args.start)
end_seconds = parse_time(args.end)
duration_seconds = end_seconds - start_seconds

if duration_seconds <= 0:
    print(f"ERROR: End time must be after start time")
    sys.exit(1)

# Set resolution - MUST match calibration canvas!
# Calibration uses court_image.png (7341x4261 horizontal, 4261x7341 after 90° CW rotation)
if args.resolution == "highres":
    HORIZONTAL_WIDTH = 7341  # Court image horizontal width
    HORIZONTAL_HEIGHT = 4261  # Court image horizontal height
    res_label = "highres"
else:
    # Compact mode for UWB-only (no calibration)
    HORIZONTAL_WIDTH = 1200
    HORIZONTAL_HEIGHT = 800
    res_label = "compact"

# For calibration-based projection, FORCE high-res
if args.resolution == "compact":
    print("⚠️  WARNING: Compact mode not compatible with calibration!")
    print("   Switching to high-res to match calibration canvas (7341x4261)")
    HORIZONTAL_WIDTH = 7341
    HORIZONTAL_HEIGHT = 4261
    res_label = "highres_forced"

# Output filenames
if args.output:
    OUTPUT_VIDEO = args.output
else:
    start_label = args.start.replace(':', '')
    end_label = args.end.replace(':', '')
    OUTPUT_VIDEO = f"red_dots_{start_label}_to_{end_label}_{res_label}.mp4"

if args.coords_log:
    OUTPUT_COORDS_LOG = args.coords_log
else:
    start_label = args.start.replace(':', '')
    end_label = args.end.replace(':', '')
    OUTPUT_COORDS_LOG = f"red_coords_{start_label}_to_{end_label}.json"

# ============================================================================
# CONFIGURATION
# ============================================================================

COURT_IMAGE = Path("data/calibration/court_image.png")
FPS = 29.97

print(f"\n{'='*70}")
print(f"Red Dots Video Generator (with Coordinate Logging)")
print(f"{'='*70}")
print(f"Video: {args.video}")
print(f"Resolution: {res_label} ({HORIZONTAL_WIDTH}x{HORIZONTAL_HEIGHT})")
print(f"Time window: {args.start} to {args.end} from video start")
print(f"Duration: {duration_seconds / 60:.1f} minutes")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"Output coords: {OUTPUT_COORDS_LOG}")
print(f"{'='*70}\n")

# Load court image (what calibration was done on)
print(f"Loading court image: {COURT_IMAGE}")
court_canvas_orig = cv2.imread(str(COURT_IMAGE))
if court_canvas_orig is None:
    print(f"ERROR: Could not load court image from {COURT_IMAGE}")
    sys.exit(1)

orig_height, orig_width = court_canvas_orig.shape[:2]
print(f"Original court image: {orig_width}x{orig_height}")

# CRITICAL: Rotate 90° clockwise FIRST - calibration was done on rotated image!
court_canvas_template = cv2.rotate(court_canvas_orig, cv2.ROTATE_90_CLOCKWISE)
rotated_height, rotated_width = court_canvas_template.shape[:2]
print(f"Rotated court image 90° CW: {rotated_width}x{rotated_height}")
print(f"Canvas dimensions: {rotated_width}x{rotated_height}")

# Update canvas dimensions to match rotated image
CANVAS_WIDTH = rotated_width   # After rotation: 4261
CANVAS_HEIGHT = rotated_height  # After rotation: 7341

# Initialize player detector and calibration
print("\nInitializing player detection...")
player_detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.3)
calibration = CalibrationIntegration(args.calibration)

# Open video
print(f"\nOpening video: {args.video}")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"ERROR: Could not open video {args.video}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames at {video_fps} fps")

# Calculate frame range
start_frame = int(start_seconds * video_fps)
end_frame = int(end_seconds * video_fps)
num_frames = end_frame - start_frame

print(f"Processing frames {start_frame} to {end_frame} ({num_frames} frames)")

# Set start position
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Create video writer (output is vertical, already rotated)
# Calculate dot size based on resolution (scale with court size) - Match blue dots!
DOT_RADIUS = int(40 * (HORIZONTAL_WIDTH / 7341))  # Scale with width
DOT_OUTLINE = max(2, int(DOT_RADIUS / 5))  # Thicker outline
print(f"Player dot radius: {DOT_RADIUS}px, outline: {DOT_OUTLINE}px")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_fps, (CANVAS_WIDTH, CANVAS_HEIGHT))

if not out.isOpened():
    print("ERROR: Could not create video writer")
    sys.exit(1)

# Coordinate logging structure
coords_log = {
    "metadata": {
        "video_path": args.video,
        "calibration_file": args.calibration,
        "court_image": str(COURT_IMAGE),
        "start_time": args.start,
        "end_time": args.end,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": video_fps,
        "canvas_size": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "note": "Canvas is already rotated 90° CW from original court image"
        },
        "generated_at": datetime.now().isoformat()
    },
    "frames": []
}

# ============================================================================
# PROCESS FRAMES
# ============================================================================

print(f"\nProcessing {num_frames} frames...")

for frame_idx in tqdm(range(num_frames)):
    frame_number = start_frame + frame_idx
    video_time = frame_number / video_fps

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print(f"\nEnd of video at frame {frame_number}")
        break

    # Use court image as canvas
    canvas = court_canvas_template.copy()

    # Detect players
    detections = player_detector.detect_players(frame)

    # Frame coordinate data
    frame_data = {
        "frame_number": frame_number,
        "video_time": video_time,
        "players": []
    }

    for detection in detections:
        bx, by = detection['bottom']

        try:
            # Homography returns coordinates in ROTATED (vertical) court canvas space
            # Canvas is already rotated 90° CW, so these coordinates are directly usable!
            canvas_x, canvas_y = calibration.image_to_court(bx, by)
            canvas_x, canvas_y = int(canvas_x), int(canvas_y)

            # Log coordinates (in vertical canvas space)
            player_data = {
                "video_pixel": [int(bx), int(by)],
                "canvas_vertical_pixels": [canvas_x, canvas_y],
                "confidence": float(detection['confidence'])
            }
            frame_data["players"].append(player_data)

            # Check bounds and draw on vertical canvas (already rotated)
            if 0 <= canvas_x < CANVAS_WIDTH and 0 <= canvas_y < CANVAS_HEIGHT:
                cv2.circle(canvas, (canvas_x, canvas_y), DOT_RADIUS, (0, 0, 255), -1)  # Red filled
                cv2.circle(canvas, (canvas_x, canvas_y), DOT_RADIUS, (255, 255, 255), DOT_OUTLINE)  # White outline
        except Exception as e:
            frame_data["players"].append({
                "video_pixel": [int(bx), int(by)],
                "error": str(e),
                "confidence": float(detection['confidence'])
            })

    coords_log["frames"].append(frame_data)

    # Canvas is already in vertical orientation - write directly!
    out.write(canvas)

# ============================================================================
# CLEANUP & SAVE
# ============================================================================

cap.release()
out.release()

# Save coordinates log
print(f"\nSaving coordinates to: {OUTPUT_COORDS_LOG}")
with open(OUTPUT_COORDS_LOG, 'w') as f:
    json.dump(coords_log, f, indent=2)

# Summary
total_players = sum(len(f["players"]) for f in coords_log["frames"])
avg_players_per_frame = total_players / len(coords_log["frames"]) if coords_log["frames"] else 0

print(f"\n✅ Video saved to: {OUTPUT_VIDEO}")
print(f"✅ Coordinates saved to: {OUTPUT_COORDS_LOG}")
print(f"\nStats:")
print(f"  Frames processed: {len(coords_log['frames'])}")
print(f"  Total player detections: {total_players}")
print(f"  Avg players per frame: {avg_players_per_frame:.1f}")
print(f"  Duration: {duration_seconds / 60:.1f} minutes")
print(f"  Resolution: {CANVAS_WIDTH}x{CANVAS_HEIGHT} (vertical, rotated 90° CW from original)")
