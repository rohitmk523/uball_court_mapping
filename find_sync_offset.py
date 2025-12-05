#!/usr/bin/env python3
"""
Find optimal sync offset between video (red dots) and UWB log (blue dots).

Strategy:
1. Load red dot coordinates from 15-20 min video segment
2. Load ENTIRE UWB log (all tags, full duration)
3. Slide the red dots through the UWB timeline
4. For each offset, calculate how well red dots match blue dots
5. Report best offset (minimum average distance)

Usage: python find_sync_offset.py
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from bisect import bisect_left

import sys
sys.path.insert(0, str(Path(__file__).parent))
from app.services.dxf_parser import parse_court_dxf

# ============================================================================
# CONFIGURATION
# ============================================================================

RED_COORDS_JSON = "red_coords_15to20min.json"
TAGS_DIR = Path("data/tags")
COURT_DXF = Path("court_2.dxf")

# Video segment info
VIDEO_START_OFFSET = 15 * 60  # 15 minutes in seconds (where video segment starts)
VIDEO_DURATION = 5 * 60  # 5 minutes in seconds
FPS = 29.97

# Search parameters
SEARCH_WINDOW_START = 0  # Start searching from beginning of UWB log
SEARCH_WINDOW_END = 79 * 60  # Search through entire UWB log (79 minutes)
SEARCH_STEP = 5  # Try offsets every 5 seconds
MAX_MATCH_DISTANCE = 500  # 500cm = 5m max distance to consider a match

# Court canvas dimensions (horizontal, will match red coords transformation)
HORIZONTAL_WIDTH = 7341
HORIZONTAL_HEIGHT = 4261

print(f"\n{'='*70}")
print(f"Sync Offset Finder: Video (Red) vs UWB (Blue)")
print(f"{'='*70}")
print(f"Red coords: {RED_COORDS_JSON}")
print(f"UWB tags: {TAGS_DIR}")
print(f"Search window: {SEARCH_WINDOW_START}s to {SEARCH_WINDOW_END}s")
print(f"Search step: {SEARCH_STEP}s")
print(f"{'='*70}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

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

def worldToScreen(x, y):
    """Convert UWB world coordinates (cm) to horizontal canvas pixels"""
    sx = ((x - courtBounds.min_x) * actualScale) + offsetX
    sy = HORIZONTAL_HEIGHT - (((y - courtBounds.min_y) * actualScale) + offsetY)
    return (int(sx), int(sy))

def screenToWorld(sx, sy):
    """Convert horizontal canvas pixels to UWB world coordinates (cm)"""
    x = ((sx - offsetX) / actualScale) + courtBounds.min_x
    y = ((HORIZONTAL_HEIGHT - sy - offsetY) / actualScale) + courtBounds.min_y
    return (x, y)

# Load red dot coordinates (player detections)
print(f"\nLoading red dot coordinates: {RED_COORDS_JSON}")
with open(RED_COORDS_JSON, 'r') as f:
    red_coords_data = json.load(f)

print(f"Loaded {len(red_coords_data['frames'])} frames of red dot data")

# Convert red dot pixel coordinates to world coordinates for comparison
print("Converting red dots from vertical canvas to world coordinates...")
red_dots_world = []  # List of (frame_idx, [(x_cm, y_cm), ...])

for frame_data in red_coords_data['frames']:
    frame_idx = frame_data['frame_number']
    players_world = []

    for player in frame_data.get('players', []):
        if 'canvas_vertical_pixels' not in player:
            continue

        # Red dots are in VERTICAL canvas coordinates
        # Convert to horizontal first (reverse 90° CW rotation)
        px_vert, py_vert = player['canvas_vertical_pixels']
        px_horiz = py_vert
        py_horiz = HORIZONTAL_HEIGHT - 1 - px_vert

        # Convert horizontal canvas pixels to world coordinates
        x_cm, y_cm = screenToWorld(px_horiz, py_horiz)
        players_world.append((x_cm, y_cm))

    if players_world:  # Only include frames with detected players
        red_dots_world.append((frame_idx, players_world))

print(f"Converted {len(red_dots_world)} frames with player detections")
print(f"Total player detections: {sum(len(p) for _, p in red_dots_world)}")

# Load UWB tag data (all tags, full timeline)
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

# Pre-sort UWB positions by timestamp for fast lookup
print("Pre-sorting UWB positions...")
sorted_tag_data = {}
for tag_id, positions in tag_data.items():
    sorted_positions = []
    for pos in positions:
        dt = datetime.fromisoformat(pos['datetime'])
        sorted_positions.append((dt, pos['x'], pos['y']))
    sorted_positions.sort(key=lambda x: x[0])
    sorted_tag_data[tag_id] = sorted_positions

print(f"Sorted {sum(len(v) for v in sorted_tag_data.values())} UWB positions")

# ============================================================================
# SYNC SEARCH
# ============================================================================

def get_uwb_positions_at_time(target_seconds):
    """Get all UWB tag positions at a specific time offset from UWB log start"""
    target_dt = uwb_start + timedelta(seconds=target_seconds)

    uwb_positions = []
    for tag_id, sorted_positions in sorted_tag_data.items():
        if not sorted_positions:
            continue

        timestamps = [dt for dt, x, y in sorted_positions]
        idx = bisect_left(timestamps, target_dt)

        # Find closest position within 5 seconds
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs((timestamps[idx - 1] - target_dt).total_seconds())))
        if idx < len(timestamps):
            candidates.append((idx, abs((timestamps[idx] - target_dt).total_seconds())))

        if candidates:
            best_idx, time_diff = min(candidates, key=lambda x: x[1])
            if time_diff < 5.0:
                dt, x, y = sorted_positions[best_idx]
                uwb_positions.append((x, y))

    return uwb_positions

def calculate_match_score(red_positions, uwb_positions):
    """
    Calculate how well red dots match blue dots.
    Returns: (avg_distance, match_count, total_distance)

    - avg_distance: average minimum distance from each red dot to nearest blue dot
    - match_count: number of red dots within MAX_MATCH_DISTANCE of a blue dot
    - total_distance: sum of all minimum distances
    """
    if not red_positions or not uwb_positions:
        return (float('inf'), 0, float('inf'))

    total_distance = 0
    match_count = 0

    for rx, ry in red_positions:
        # Find nearest UWB tag
        min_dist = float('inf')
        for ux, uy in uwb_positions:
            dist = np.sqrt((rx - ux)**2 + (ry - uy)**2)
            min_dist = min(min_dist, dist)

        total_distance += min_dist
        if min_dist <= MAX_MATCH_DISTANCE:
            match_count += 1

    avg_distance = total_distance / len(red_positions)
    return (avg_distance, match_count, total_distance)

print(f"\nSearching for best sync offset...")
print(f"Testing offsets from {SEARCH_WINDOW_START}s to {SEARCH_WINDOW_END}s (step {SEARCH_STEP}s)")

# Sample frames for faster search (every 30 frames = ~1 second)
sample_frames = red_dots_world[::30]
print(f"Using {len(sample_frames)} sample frames for speed")

# Try different offsets
results = []
offsets_to_test = range(SEARCH_WINDOW_START, SEARCH_WINDOW_END, SEARCH_STEP)

for uwb_offset in tqdm(offsets_to_test, desc="Testing offsets"):
    # For this offset, calculate match quality across all sample frames
    frame_scores = []

    for frame_idx, red_positions in sample_frames:
        # Calculate where this video frame maps to in UWB timeline
        video_time = frame_idx / FPS  # Time within the 15-20 min segment
        uwb_time = uwb_offset + video_time  # Corresponding UWB log time

        # Get UWB positions at this time
        uwb_positions = get_uwb_positions_at_time(uwb_time)

        # Calculate match score
        avg_dist, match_count, total_dist = calculate_match_score(red_positions, uwb_positions)
        frame_scores.append((avg_dist, match_count, total_dist))

    # Aggregate scores across all frames
    if frame_scores:
        avg_distances = [s[0] for s in frame_scores]
        match_counts = [s[1] for s in frame_scores]
        total_distances = [s[2] for s in frame_scores]

        overall_avg_dist = np.mean(avg_distances)
        overall_match_count = np.sum(match_counts)
        overall_total_dist = np.sum(total_distances)

        results.append({
            'offset': uwb_offset,
            'avg_distance': overall_avg_dist,
            'match_count': overall_match_count,
            'total_distance': overall_total_dist,
            'match_rate': overall_match_count / sum(len(p) for _, p in sample_frames)
        })

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n{'='*70}")
print(f"SYNC OFFSET ANALYSIS RESULTS")
print(f"{'='*70}\n")

# Sort by average distance (lower is better)
results_sorted = sorted(results, key=lambda x: x['avg_distance'])

print("Top 10 Best Sync Offsets (by average distance):")
print(f"\n{'Rank':<6}{'Offset':<12}{'Avg Dist':<12}{'Matches':<10}{'Match Rate':<12}")
print(f"{'-'*70}")

for i, result in enumerate(results_sorted[:10], 1):
    offset_min = result['offset'] / 60
    avg_dist_m = result['avg_distance'] / 100  # cm to meters
    match_rate_pct = result['match_rate'] * 100

    print(f"{i:<6}{offset_min:>6.1f} min   {avg_dist_m:>6.2f} m     "
          f"{result['match_count']:<10}{match_rate_pct:>6.1f} %")

# Also sort by match count (higher is better)
results_by_matches = sorted(results, key=lambda x: x['match_count'], reverse=True)

print(f"\nTop 10 Best Sync Offsets (by match count):")
print(f"\n{'Rank':<6}{'Offset':<12}{'Avg Dist':<12}{'Matches':<10}{'Match Rate':<12}")
print(f"{'-'*70}")

for i, result in enumerate(results_by_matches[:10], 1):
    offset_min = result['offset'] / 60
    avg_dist_m = result['avg_distance'] / 100  # cm to meters
    match_rate_pct = result['match_rate'] * 100

    print(f"{i:<6}{offset_min:>6.1f} min   {avg_dist_m:>6.2f} m     "
          f"{result['match_count']:<10}{match_rate_pct:>6.1f} %")

# Best overall offset
best_offset = results_sorted[0]
best_offset_sec = best_offset['offset']
best_offset_min = best_offset_sec / 60

print(f"\n{'='*70}")
print(f"RECOMMENDED SYNC OFFSET")
print(f"{'='*70}")
print(f"\nBest offset: {best_offset_min:.2f} minutes ({best_offset_sec} seconds)")
print(f"Average distance: {best_offset['avg_distance'] / 100:.2f} meters")
print(f"Match count: {best_offset['match_count']} / {sum(len(p) for _, p in sample_frames)} "
      f"({best_offset['match_rate'] * 100:.1f}%)")
print(f"\nInterpretation:")
print(f"  - Video segment starts at 15:00 (900 seconds from video start)")
print(f"  - This corresponds to UWB time: {best_offset_min:.2f} minutes from UWB log start")
print(f"  - Video-to-UWB offset: {best_offset_sec - 900:.0f} seconds")
print(f"    (i.e., video is {abs(best_offset_sec - 900):.0f}s {'ahead' if best_offset_sec > 900 else 'behind'} UWB log)")

# Save results
output_file = "sync_offset_analysis.json"
with open(output_file, 'w') as f:
    json.dump({
        'best_offset_seconds': best_offset_sec,
        'best_offset_minutes': best_offset_min,
        'avg_distance_cm': best_offset['avg_distance'],
        'match_count': best_offset['match_count'],
        'match_rate': best_offset['match_rate'],
        'all_results': results_sorted[:20]  # Top 20 results
    }, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
