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
from bisect import bisect_left

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

# Set resolution - MUST match court image!
# Court image is 7341x4261 (horizontal), will be rotated 90° CW for output
if args.resolution == "highres":
    HORIZONTAL_WIDTH = 7341  # Match court image width
    HORIZONTAL_HEIGHT = 4261  # Match court image height
    res_label = "highres"
else:
    # Force high-res to match court image
    print("⚠️  WARNING: Forcing high-res to match court image!")
    HORIZONTAL_WIDTH = 7341
    HORIZONTAL_HEIGHT = 4261
    res_label = "highres_forced"

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
COURT_IMAGE = Path("data/calibration/court_image.png")
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

# Load pre-rendered court image (FAST - no DXF parsing needed!)
print(f"Loading court image: {COURT_IMAGE}")
court_canvas_horizontal = cv2.imread(str(COURT_IMAGE))
if court_canvas_horizontal is None:
    print(f"ERROR: Could not load court image from {COURT_IMAGE}")
    sys.exit(1)

img_height, img_width = court_canvas_horizontal.shape[:2]
print(f"Court image loaded: {img_width}x{img_height}")

# Verify dimensions match
if img_width != HORIZONTAL_WIDTH or img_height != HORIZONTAL_HEIGHT:
    print(f"WARNING: Court image size ({img_width}x{img_height}) doesn't match canvas size ({HORIZONTAL_WIDTH}x{HORIZONTAL_HEIGHT})")
    print(f"Resizing court image to match canvas...")
    court_canvas_horizontal = cv2.resize(court_canvas_horizontal, (HORIZONTAL_WIDTH, HORIZONTAL_HEIGHT))

# Load court geometry (needed for UWB coordinate transformation)
print("Loading court geometry for coordinate transformation...")
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

# OPTIMIZED: Pre-sort positions by timestamp and use binary search
print("Pre-sorting UWB positions by timestamp...")
sorted_tag_data = {}
for tag_id, positions in tag_data.items():
    # Convert timestamps to datetime objects and sort
    sorted_positions = []
    for pos in positions:
        dt = datetime.fromisoformat(pos['datetime'])
        sorted_positions.append((dt, pos['x'], pos['y']))
    sorted_positions.sort(key=lambda x: x[0])  # Sort by datetime
    sorted_tag_data[tag_id] = sorted_positions

print(f"Sorted {len(sorted_tag_data)} tags with {sum(len(v) for v in sorted_tag_data.values())} total positions")

# Build optimized frame-to-position index using BINARY SEARCH (100x faster!)
print("Building frame index using binary search...")
frame_tag_positions = {}  # {frame_idx: {tag_id: (x, y)}}

for frame_idx in range(num_frames):
    current_seconds = start_seconds + (frame_idx / FPS)
    target_dt = uwb_start + timedelta(seconds=current_seconds)

    frame_tag_positions[frame_idx] = {}

    for tag_id, sorted_positions in sorted_tag_data.items():
        if not sorted_positions:
            continue

        # Binary search for closest timestamp
        timestamps = [dt for dt, x, y in sorted_positions]
        idx = bisect_left(timestamps, target_dt)

        # Check adjacent positions to find the closest one
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs((timestamps[idx - 1] - target_dt).total_seconds())))
        if idx < len(timestamps):
            candidates.append((idx, abs((timestamps[idx] - target_dt).total_seconds())))

        if candidates:
            best_idx, time_diff = min(candidates, key=lambda x: x[1])
            if time_diff < 5.0:  # Within 5 seconds threshold
                dt, x, y = sorted_positions[best_idx]
                frame_tag_positions[frame_idx][tag_id] = (x, y)

print(f"Index built for {num_frames} frames in seconds (not minutes!)")

# Calculate dot size based on resolution (scale with court size)
# Red dots use radius 20 at this resolution, blue dots should be similar
DOT_RADIUS = int(20 * (HORIZONTAL_WIDTH / 7341))  # Scale with width
DOT_OUTLINE = max(2, int(DOT_RADIUS / 10))  # Proportional outline
print(f"UWB dot radius: {DOT_RADIUS}px, outline: {DOT_OUTLINE}px")

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
    # Use pre-rendered court image as canvas (FAST!)
    canvas = court_canvas_horizontal.copy()

    # Draw UWB tags using pre-built index (SUPER FAST - O(1) lookup!)
    if frame_idx in frame_tag_positions:
        for tag_id, (x_uwb, y_uwb) in frame_tag_positions[frame_idx].items():
            sx, sy = worldToScreen(x_uwb, y_uwb)
            # Draw blue dot with white outline (scaled for resolution)
            cv2.circle(canvas, (sx, sy), DOT_RADIUS, (255, 100, 0), -1)  # Blue filled
            cv2.circle(canvas, (sx, sy), DOT_RADIUS, (255, 255, 255), DOT_OUTLINE)  # White outline

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
