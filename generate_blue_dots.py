#!/usr/bin/env python3
"""
Blue dots video generator: UWB tags only - No player detection.
Usage: python generate_blue_dots.py --start 00:25:00 --end 00:26:00 --resolution compact
"""
import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from app.services.dxf_parser import parse_court_dxf

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description='Generate blue dots (UWB tags only) court video',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  # 1 minute from 25-26 min, compact resolution
  python generate_blue_dots.py --start 00:25:00 --end 00:26:00 --resolution compact

  # 5 minutes from 30-35 min, high resolution
  python generate_blue_dots.py --start 00:30:00 --end 00:35:00 --resolution highres

  # Full UWB log duration
  python generate_blue_dots.py --start 00:00:00 --end 01:00:00
'''
)

parser.add_argument(
    '--start',
    type=str,
    required=True,
    help='Start time in HH:MM:SS format (from UWB log start)'
)

parser.add_argument(
    '--end',
    type=str,
    required=True,
    help='End time in HH:MM:SS format (from UWB log start)'
)

parser.add_argument(
    '--resolution',
    type=str,
    choices=['compact', 'highres'],
    default='compact',
    help='Canvas resolution: compact (800x1200) or highres (4260x7340)'
)

parser.add_argument(
    '--output',
    type=str,
    default=None,
    help='Output video filename (default: auto-generated)'
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

# Set resolution
if args.resolution == "highres":
    HORIZONTAL_WIDTH = 7340
    HORIZONTAL_HEIGHT = 4260
    res_label = "highres"
else:
    HORIZONTAL_WIDTH = 1200
    HORIZONTAL_HEIGHT = 800
    res_label = "compact"

# Output filename
if args.output:
    OUTPUT_VIDEO = args.output
else:
    start_label = args.start.replace(':', '')
    end_label = args.end.replace(':', '')
    OUTPUT_VIDEO = f"blue_dots_{start_label}_to_{end_label}_{res_label}.mp4"

# ============================================================================
# CONFIGURATION
# ============================================================================

TAGS_DIR = Path("data/tags")
COURT_DXF = Path("court_2.dxf")
FPS = 29.97
num_frames = int(duration_seconds * FPS)

print(f"\n{'='*70}")
print(f"Blue Dots Video Generator")
print(f"{'='*70}")
print(f"Resolution: {res_label} ({HORIZONTAL_WIDTH}x{HORIZONTAL_HEIGHT})")
print(f"Time window: {args.start} to {args.end} from UWB log start")
print(f"Duration: {duration_seconds / 60:.1f} minutes ({num_frames} frames)")
print(f"Output: {OUTPUT_VIDEO}")
print(f"{'='*70}\n")

# Load court geometry
print("Loading court geometry...")
geometry = parse_court_dxf(COURT_DXF)
courtBounds = geometry.bounds

print(f"Court bounds: X({courtBounds.min_x:.1f} to {courtBounds.max_x:.1f}), "
      f"Y({courtBounds.min_y:.1f} to {courtBounds.max_y:.1f})")

# Calculate scale
courtWidth = courtBounds.max_x - courtBounds.min_x
courtHeight = courtBounds.max_y - courtBounds.min_y

PADDING = 50
scaleX = (HORIZONTAL_WIDTH - PADDING * 2) / courtWidth
scaleY = (HORIZONTAL_HEIGHT - PADDING * 2) / courtHeight
actualScale = min(scaleX, scaleY)

scaledWidth = courtWidth * actualScale
scaledHeight = courtHeight * actualScale
offsetX = (HORIZONTAL_WIDTH - scaledWidth) / 2
offsetY = (HORIZONTAL_HEIGHT - scaledHeight) / 2

print(f"Scale: {actualScale:.6f}")

# World to screen transformation
def worldToScreen(x, y):
    sx = ((x - courtBounds.min_x) * actualScale) + offsetX
    sy = HORIZONTAL_HEIGHT - (((y - courtBounds.min_y) * actualScale) + offsetY)
    return (int(sx), int(sy))

# Load UWB tag data
print("\nLoading UWB tag data...")
tag_data = {}
for tag_file in TAGS_DIR.glob('*.json'):
    with open(tag_file, 'r') as f:
        data = json.load(f)
        tag_data[data['tag_id']] = data['positions']

print(f"Loaded {len(tag_data)} tags")

# Get UWB log start time
first_tag_id = list(tag_data.keys())[0]
uwb_start = datetime.fromisoformat(tag_data[first_tag_id][0]['datetime'])
print(f"UWB log starts at: {uwb_start}")

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (HORIZONTAL_HEIGHT, HORIZONTAL_WIDTH))

if not out.isOpened():
    print("ERROR: Could not create video writer")
    sys.exit(1)

# ============================================================================
# GENERATE FRAMES
# ============================================================================

print(f"\nProcessing {num_frames} frames...")

for frame_idx in tqdm(range(num_frames)):
    current_seconds = start_seconds + (frame_idx / FPS)
    target_dt = uwb_start + timedelta(seconds=current_seconds)

    # Create horizontal canvas
    canvas = np.zeros((HORIZONTAL_HEIGHT, HORIZONTAL_WIDTH, 3), dtype=np.uint8)

    # Draw court
    for line in geometry.lines:
        pt1 = worldToScreen(*line[0])
        pt2 = worldToScreen(*line[1])
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 1)

    for polyline in geometry.polylines:
        pts = np.array([worldToScreen(x, y) for x, y in polyline], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)

    for circle in geometry.circles:
        if circle[1] > 300:
            continue
        center = worldToScreen(*circle[0])
        radius = int(circle[1] * actualScale)
        cv2.circle(canvas, center, radius, (255, 255, 255), 1)

    # Draw UWB tags (BLUE)
    for tag_id, positions in tag_data.items():
        closest_pos = None
        min_diff = float('inf')

        for pos in positions:
            pos_time = datetime.fromisoformat(pos['datetime'])
            time_diff = abs((pos_time - target_dt).total_seconds())
            if time_diff < min_diff and time_diff < 5.0:
                min_diff = time_diff
                closest_pos = pos

        if closest_pos:
            x_uwb = closest_pos['x']
            y_uwb = closest_pos['y']
            sx, sy = worldToScreen(x_uwb, y_uwb)
            cv2.circle(canvas, (sx, sy), 8, (255, 100, 0), -1)  # Blue
            cv2.circle(canvas, (sx, sy), 8, (255, 255, 255), 1)  # White outline

    # Rotate 90° clockwise for vertical output
    canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    out.write(canvas_rotated)

# ============================================================================
# CLEANUP
# ============================================================================

out.release()

print(f"\n✅ Video saved to: {OUTPUT_VIDEO}")
print(f"Duration: {duration_seconds / 60:.1f} minutes")
print(f"Resolution: {HORIZONTAL_HEIGHT}x{HORIZONTAL_WIDTH} (vertical)")
