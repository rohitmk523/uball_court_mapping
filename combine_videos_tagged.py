#!/usr/bin/env python3
"""
Enhanced video combiner with tagging logic:
- Blue dots (UWB tags) with 200cm radius circles
- Red dots (player detections) turn GREEN when within 200cm of a tag
- Green dots display the tag_id

Usage: python combine_videos_tagged.py <start_time> <end_time> <output_suffix>
Example: python combine_videos_tagged.py 00:15:00 00:20:00 v3
"""
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from bisect import bisect_left

sys.path.insert(0, str(Path(__file__).parent))
from app.services.dxf_parser import parse_court_dxf

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

import argparse

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Combine red and blue dot videos with tagging logic')
parser.add_argument('start_time', help='Start time in HH:MM:SS')
parser.add_argument('end_time', help='End time in HH:MM:SS')
parser.add_argument('red_coords_json', help='Path to red dot coordinates JSON file')
parser.add_argument('output_suffix', help='Suffix for output filename')
parser.add_argument('--sync-offset', type=float, default=785.0, help='Sync offset in seconds (video ahead of UWB). Default: 785.0')

args = parser.parse_args()

START_TIME = args.start_time
END_TIME = args.end_time
RED_COORDS_JSON = args.red_coords_json
SUFFIX = args.output_suffix
UWB_SYNC_OFFSET = args.sync_offset

def parse_time(time_str):
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

start_seconds = parse_time(START_TIME)
end_seconds = parse_time(END_TIME)
duration_seconds = end_seconds - start_seconds

if duration_seconds <= 0:
    print(f"ERROR: End time must be after start time")
    sys.exit(1)

# Generate output filename from red coords JSON filename
# Extract the time label from the red coords JSON filename
# e.g., "red_coords_15to20min.json" -> "15to20min"
coords_basename = Path(RED_COORDS_JSON).stem  # Remove .json extension
time_label = coords_basename.replace("red_coords_", "")  # Remove prefix

TAGS_DIR = Path("data/tags")
COURT_DXF = Path("court_2.dxf")
COURT_IMAGE = Path("data/calibration/court_image.png")
OUTPUT_VIDEO = f"combined_{time_label}_{SUFFIX}.mp4"

FPS = 29.97
num_frames = int(duration_seconds * FPS)

print(f"\n{'='*70}")
print(f"Tagged Video Combiner: Red + Blue Dots with Tagging")
print(f"{'='*70}")
print(f"Time window: {START_TIME} to {END_TIME}")
print(f"Duration: {duration_seconds / 60:.1f} minutes ({num_frames} frames)")
print(f"Red coords: {RED_COORDS_JSON}")
print(f"Output: {OUTPUT_VIDEO}")
print(f"{'='*70}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

# Load court image (horizontal, will be rotated 90° CW)
print(f"Loading court image: {COURT_IMAGE}")
court_canvas_horizontal = cv2.imread(str(COURT_IMAGE))
if court_canvas_horizontal is None:
    print(f"ERROR: Could not load court image from {COURT_IMAGE}")
    sys.exit(1)

img_height, img_width = court_canvas_horizontal.shape[:2]
print(f"Court image loaded: {img_width}x{img_height}")

HORIZONTAL_WIDTH = img_width
HORIZONTAL_HEIGHT = img_height

# Load court geometry for coordinate transformation
print("Loading court geometry...")
geometry = parse_court_dxf(COURT_DXF)
courtBounds = geometry.bounds

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

print(f"Scale: {actualScale:.6f} pixels/cm")
print(f"200cm radius = {200 * actualScale:.1f} pixels")

# World to screen transformation (for UWB tags)
def worldToScreen(x, y):
    """Convert UWB world coordinates (cm) to horizontal canvas pixels"""
    sx = ((x - courtBounds.min_x) * actualScale) + offsetX
    sy = HORIZONTAL_HEIGHT - (((y - courtBounds.min_y) * actualScale) + offsetY)
    return (int(sx), int(sy))

# Load red dot coordinates
print(f"\nLoading red dot coordinates: {RED_COORDS_JSON}")
if not Path(RED_COORDS_JSON).exists():
    print(f"ERROR: Red coordinates file not found: {RED_COORDS_JSON}")
    sys.exit(1)

with open(RED_COORDS_JSON, 'r') as f:
    red_coords_data = json.load(f)

print(f"Loaded {len(red_coords_data['frames'])} frames of red dot data")

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

# Pre-sort and index UWB positions
print("Pre-sorting UWB positions by timestamp...")
sorted_tag_data = {}
for tag_id, positions in tag_data.items():
    sorted_positions = []
    for pos in positions:
        dt = datetime.fromisoformat(pos['datetime'])
        sorted_positions.append((dt, pos['x'], pos['y']))
    sorted_positions.sort(key=lambda x: x[0])
    sorted_tag_data[tag_id] = sorted_positions

# Sync offset is now loaded from arguments
# UWB_SYNC_OFFSET = args.sync_offset

# Build frame-to-UWB-position index
print("Building UWB frame index...")
print(f"Applying sync offset: {UWB_SYNC_OFFSET}s ({UWB_SYNC_OFFSET/60:.1f} min)")
frame_uwb_positions = {}

for frame_idx in range(num_frames):
    current_seconds = start_seconds + (frame_idx / FPS)
    # Apply sync offset: video time + offset = UWB time
    target_dt = uwb_start + timedelta(seconds=current_seconds + UWB_SYNC_OFFSET)

    frame_uwb_positions[frame_idx] = {}

    for tag_id, sorted_positions in sorted_tag_data.items():
        if not sorted_positions:
            continue

        timestamps = [dt for dt, x, y in sorted_positions]
        idx = bisect_left(timestamps, target_dt)

        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs((timestamps[idx - 1] - target_dt).total_seconds())))
        if idx < len(timestamps):
            candidates.append((idx, abs((timestamps[idx] - target_dt).total_seconds())))

        if candidates:
            best_idx, time_diff = min(candidates, key=lambda x: x[1])
            if time_diff < 5.0:
                dt, x, y = sorted_positions[best_idx]
                frame_uwb_positions[frame_idx][tag_id] = (x, y)

print(f"UWB index built for {num_frames} frames")

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

DOT_RADIUS = 20
DOT_OUTLINE = 2
RADIUS_200CM = int(200 * actualScale)  # 200cm in pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 3

print(f"\nVisualization parameters:")
print(f"  Dot radius: {DOT_RADIUS}px")
print(f"  200cm radius circle: {RADIUS_200CM}px")
print(f"  Font scale: {FONT_SCALE}")

# ============================================================================
# CREATE VIDEO
# ============================================================================

# Create video writer (output is vertical after rotation)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (HORIZONTAL_HEIGHT, HORIZONTAL_WIDTH))

if not out.isOpened():
    print("ERROR: Could not create video writer")
    sys.exit(1)

print(f"\nGenerating {num_frames} frames...")

for frame_idx in tqdm(range(num_frames)):
    # Start with court canvas
    canvas = court_canvas_horizontal.copy()

    # Get UWB tag positions for this frame (in horizontal canvas)
    uwb_positions = frame_uwb_positions.get(frame_idx, {})

    # Get red dot positions for this frame
    if frame_idx < len(red_coords_data['frames']):
        frame_data = red_coords_data['frames'][frame_idx]
        red_players = frame_data.get('players', [])
    else:
        red_players = []

    # Convert UWB positions to horizontal canvas coordinates
    uwb_screen_positions = {}
    for tag_id, (x_uwb, y_uwb) in uwb_positions.items():
        sx, sy = worldToScreen(x_uwb, y_uwb)
        uwb_screen_positions[tag_id] = (sx, sy)

    # Draw 200cm radius circles around UWB tags (semi-transparent)
    overlay = canvas.copy()
    for tag_id, (sx, sy) in uwb_screen_positions.items():
        cv2.circle(overlay, (sx, sy), RADIUS_200CM, (255, 200, 0), 2)  # Light blue circle
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    # Draw UWB tags (blue dots)
    for tag_id, (sx, sy) in uwb_screen_positions.items():
        cv2.circle(canvas, (sx, sy), DOT_RADIUS, (255, 100, 0), -1)  # Blue filled
        cv2.circle(canvas, (sx, sy), DOT_RADIUS, (255, 255, 255), DOT_OUTLINE)  # White outline

    # Process red dots - check distance to UWB tags
    for player in red_players:
        if 'canvas_vertical_pixels' not in player:
            continue

        # Red dot is in VERTICAL canvas coordinates (after 90° CW rotation)
        # Need to convert back to horizontal for distance calculation
        px_vert, py_vert = player['canvas_vertical_pixels']

        # Reverse the 90° CW rotation: vertical (4261x7341) -> horizontal (7341x4261)
        # OpenCV's ROTATE_90_CLOCKWISE maps (x_h, y_h) -> (H-1-y_h, x_h)
        # To reverse: (x_v, y_v) -> (y_v, H-1-x_v)
        # Where H is the horizontal canvas height (4261)
        px_horiz = py_vert
        py_horiz = HORIZONTAL_HEIGHT - 1 - px_vert

        # Check distance to each UWB tag
        closest_tag_id = None
        min_distance = float('inf')

        for tag_id, (sx, sy) in uwb_screen_positions.items():
            distance = np.sqrt((px_horiz - sx)**2 + (py_horiz - sy)**2)
            if distance < min_distance:
                min_distance = distance
                closest_tag_id = tag_id

        # Determine color based on distance
        if closest_tag_id and min_distance <= RADIUS_200CM:
            # GREEN - tagged player
            color = (0, 255, 0)
            outline_color = (255, 255, 255)

            # Draw on horizontal canvas
            cv2.circle(canvas, (px_horiz, py_horiz), DOT_RADIUS, color, -1)
            cv2.circle(canvas, (px_horiz, py_horiz), DOT_RADIUS, outline_color, DOT_OUTLINE)

            # Add tag ID label
            label = f"Tag {closest_tag_id}"
            label_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            label_x = px_horiz - label_size[0] // 2
            label_y = py_horiz - DOT_RADIUS - 10

            # Black background for text
            cv2.rectangle(canvas,
                         (label_x - 5, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 5, label_y + 5),
                         (0, 0, 0), -1)

            # White text
            cv2.putText(canvas, label, (label_x, label_y), FONT,
                       FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        else:
            # RED - untagged player
            color = (0, 0, 255)
            outline_color = (255, 255, 255)

            # Draw on horizontal canvas
            cv2.circle(canvas, (px_horiz, py_horiz), DOT_RADIUS, color, -1)
            cv2.circle(canvas, (px_horiz, py_horiz), DOT_RADIUS, outline_color, DOT_OUTLINE)

    # Rotate 90° clockwise for vertical output
    canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    out.write(canvas_rotated)

# ============================================================================
# CLEANUP
# ============================================================================

out.release()

print(f"\n✅ Tagged video saved to: {OUTPUT_VIDEO}")
print(f"Duration: {duration_seconds / 60:.1f} minutes")
print(f"Resolution: {HORIZONTAL_HEIGHT}x{HORIZONTAL_WIDTH} (vertical)")
print(f"\nColor legend:")
print(f"  🔵 BLUE dots: UWB tags with 200cm radius circles")
print(f"  🟢 GREEN dots: Tagged players (within 200cm of UWB tag)")
print(f"  🔴 RED dots: Untagged players")
