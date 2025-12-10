#!/usr/bin/env python3
"""
Generate tracked video with persistent player IDs using ByteTrack + UWB hybrid tracking.

Features:
- Blue dots: UWB tags with 200cm radius circles
- Green dots: Tagged players (within 200cm) with UWB tag IDs
- Red dots: Untagged players with persistent track IDs
- Track IDs persist across frames using IoU matching
- Periodic UWB re-matching every 2 seconds

Usage: python generate_tracked_video.py <red_coords_json> <start_time> <end_time> <output_suffix>
Example: python generate_tracked_video.py red_coords_15to20min.json 00:15:00 00:20:00 tracked_v1
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
from app.services.player_tracker import PlayerTracker

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

import argparse

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Generate tracked video with ByteTrack + UWB hybrid tracking')
parser.add_argument('red_coords_json', help='Path to red dot coordinates JSON file')
parser.add_argument('start_time', help='Start time in HH:MM:SS')
parser.add_argument('end_time', help='End time in HH:MM:SS')
parser.add_argument('output_suffix', help='Suffix for output filename')
parser.add_argument('--sync-offset', type=float, default=785.0, help='Sync offset in seconds (video ahead of UWB). Default: 785.0')

args = parser.parse_args()

RED_COORDS_JSON = args.red_coords_json
START_TIME = args.start_time
END_TIME = args.end_time
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

# Generate output filename
coords_basename = Path(RED_COORDS_JSON).stem
time_label = coords_basename.replace("red_coords_", "")

TAGS_DIR = Path("data/tags")
COURT_DXF = Path("court_2.dxf")
COURT_IMAGE = Path("data/calibration/court_image.png")
OUTPUT_VIDEO = f"tracked_{time_label}_{SUFFIX}.mp4"

FPS = 29.97
num_frames = int(duration_seconds * FPS)

print(f"\n{'='*70}")
print(f"Tracked Video Generator: ByteTrack + UWB Hybrid Tracking")
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

# Screen to world transformation (for player positions)
def screenToWorld(sx, sy):
    """Convert horizontal canvas pixels to UWB world coordinates (cm)"""
    x = ((sx - offsetX) / actualScale) + courtBounds.min_x
    y = ((HORIZONTAL_HEIGHT - sy - offsetY) / actualScale) + courtBounds.min_y
    return (x, y)

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
# INITIALIZE TRACKER
# ============================================================================

print("\nInitializing PlayerTracker...")
tracker = PlayerTracker(
    uwb_sync_offset=UWB_SYNC_OFFSET,
    tagging_radius_cm=200.0,
    rematch_interval=2.0,
    iou_threshold=0.3,
    max_frames_lost=30
)

print("Tracker initialized with:")
print(f"  - UWB sync offset: {UWB_SYNC_OFFSET}s")
print(f"  - Tagging radius: 200cm")
print(f"  - Re-match interval: 2s")
print(f"  - IoU threshold: 0.3")
print(f"  - Max frames lost: 30 (~1 second)")

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

DOT_RADIUS = 20
DOT_OUTLINE = 2
RADIUS_200CM = int(200 * actualScale)  # 200cm in pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2

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

print(f"\nGenerating {num_frames} frames with tracking...")

# Statistics
total_players = 0
total_tagged = 0
total_tracks_created = 0

for frame_idx in tqdm(range(num_frames)):
    # Start with court canvas
    canvas = court_canvas_horizontal.copy()

    # Get video time
    current_seconds = start_seconds + (frame_idx / FPS)

    # Get UWB tag positions for this frame
    uwb_positions = frame_uwb_positions.get(frame_idx, {})

    # Get red dot positions for this frame
    if frame_idx < len(red_coords_data['frames']):
        frame_data = red_coords_data['frames'][frame_idx]
        red_players = frame_data.get('players', [])
    else:
        red_players = []

    # Convert red dots to detections format for tracker
    detections = []
    for player in red_players:
        if 'canvas_vertical_pixels' not in player:
            continue

        # Convert vertical canvas to horizontal canvas coordinates
        px_vert, py_vert = player['canvas_vertical_pixels']
        px_horiz = py_vert
        py_horiz = HORIZONTAL_HEIGHT - 1 - px_vert

        # Convert to world coordinates
        x_cm, y_cm = screenToWorld(px_horiz, py_horiz)

        # Create detection dict with bbox (approximate)
        bbox_size = 100  # pixels
        detections.append({
            'position': (x_cm, y_cm),
            'bbox': [
                px_horiz - bbox_size // 2,
                py_horiz - bbox_size // 2,
                px_horiz + bbox_size // 2,
                py_horiz + bbox_size // 2
            ]
        })

    # Update tracker
    tracked_players = tracker.update(detections, uwb_positions, current_seconds)

    # Update statistics
    total_players += len(tracked_players)
    total_tagged += sum(1 for p in tracked_players if p.uwb_tag_id is not None)
    if frame_idx == 0:
        total_tracks_created = len(tracked_players)

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

    # Draw tracked players
    for player in tracked_players:
        # Convert world position to horizontal canvas pixels
        x_cm, y_cm = player.position
        px_horiz, py_horiz = worldToScreen(x_cm, y_cm)

        # Determine color based on UWB tagging
        if player.uwb_tag_id is not None:
            # GREEN - tagged player
            color = (0, 255, 0)
            outline_color = (255, 255, 255)
            label = f"Tag {player.uwb_tag_id}"
        else:
            # RED - untagged player
            color = (0, 0, 255)
            outline_color = (255, 255, 255)
            label = f"ID {player.track_id}"

        # Draw player dot
        cv2.circle(canvas, (px_horiz, py_horiz), DOT_RADIUS, color, -1)
        cv2.circle(canvas, (px_horiz, py_horiz), DOT_RADIUS, outline_color, DOT_OUTLINE)

        # Add label
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

    # Add tracking stats to frame
    stats = tracker.get_tracking_stats()
    stats_text = f"Tracks: {stats['total_tracks']} | Tagged: {stats['tagged_tracks']} | Untagged: {stats['untagged_tracks']}"
    cv2.putText(canvas, stats_text, (20, 50), FONT, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, stats_text, (20, 50), FONT, 0.8, (0, 0, 0), 1)

    # Rotate 90° clockwise for vertical output
    canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    out.write(canvas_rotated)

# ============================================================================
# CLEANUP & REPORT
# ============================================================================

out.release()

# Calculate final statistics
avg_tagged_rate = (total_tagged / total_players * 100) if total_players > 0 else 0
final_stats = tracker.get_tracking_stats()

print(f"\n✅ Tracked video saved to: {OUTPUT_VIDEO}")
print(f"\n{'='*70}")
print(f"TRACKING STATISTICS")
print(f"{'='*70}")
print(f"Duration: {duration_seconds / 60:.1f} minutes ({num_frames} frames)")
print(f"Resolution: {HORIZONTAL_HEIGHT}x{HORIZONTAL_WIDTH} (vertical)")
print(f"\nPlayer Tracking:")
print(f"  Total player detections: {total_players}")
print(f"  Tagged players: {total_tagged} ({avg_tagged_rate:.1f}%)")
print(f"  Untagged players: {total_players - total_tagged} ({100 - avg_tagged_rate:.1f}%)")
print(f"  Unique track IDs created: {final_stats['next_track_id'] - 1}")
print(f"  Active tracks at end: {final_stats['total_tracks']}")
print(f"\nVisualization Legend:")
print(f"  🔵 BLUE dots: UWB tags with 200cm radius circles")
print(f"  🟢 GREEN dots: Tagged players with UWB tag IDs")
print(f"  🔴 RED dots: Untagged players with persistent track IDs")
print(f"\nTracking Parameters:")
print(f"  IoU threshold: 0.3 (for frame-to-frame matching)")
print(f"  UWB re-match interval: 2 seconds")
print(f"  Max frames lost: 30 frames (~1 second)")
print(f"  Tagging radius: 200cm")
