#!/usr/bin/env python3
"""
Analyze full game sync quality with red dots (players) and blue dots (UWB tags).
Reports on tagging effectiveness at 200cm radius threshold.

Usage: python analyze_full_game_sync.py
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from bisect import bisect_left
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from app.services.dxf_parser import parse_court_dxf

# ============================================================================
# CONFIGURATION
# ============================================================================

RED_COORDS_JSON = "red_coords_FULL_GAME.json"
TAGS_DIR = Path("data/tags")
COURT_DXF = Path("court_2.dxf")

# Sync offsets to test (in seconds)
OFFSETS_TO_TEST = [
    785,   # Best by avg distance (28.08 min)
    1062,  # Best by match count (17.7 min)
    1090,  # 18.2 min
    1044,  # 17.4 min
]

TAGGING_RADIUS_CM = 200  # 200cm radius for green tagging
FPS = 29.97

# Court canvas dimensions
HORIZONTAL_WIDTH = 7341
HORIZONTAL_HEIGHT = 4261

print(f"\n{'='*70}")
print(f"Full Game Sync Analysis: Red-Blue Matching & Green Tagging")
print(f"{'='*70}")
print(f"Red coords: {RED_COORDS_JSON}")
print(f"UWB tags: {TAGS_DIR}")
print(f"Tagging radius: {TAGGING_RADIUS_CM}cm")
print(f"Testing {len(OFFSETS_TO_TEST)} sync offsets")
print(f"{'='*70}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

# Load court geometry
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

# Load red dot coordinates (full game)
print(f"\nLoading red dot coordinates: {RED_COORDS_JSON}")
with open(RED_COORDS_JSON, 'r') as f:
    red_coords_data = json.load(f)

print(f"Loaded {len(red_coords_data['frames'])} frames of red dot data")

# Convert red dots to world coordinates
print("Converting red dots to world coordinates...")
red_dots_world = []  # List of (frame_idx, video_time, [(x_cm, y_cm), ...])

for frame_data in red_coords_data['frames']:
    frame_idx = frame_data['frame_number']
    video_time = frame_data['video_time']
    players_world = []

    for player in frame_data.get('players', []):
        if 'canvas_vertical_pixels' not in player:
            continue

        # Convert vertical canvas to horizontal, then to world coordinates
        px_vert, py_vert = player['canvas_vertical_pixels']
        px_horiz = py_vert
        py_horiz = HORIZONTAL_HEIGHT - 1 - px_vert

        x_cm, y_cm = screenToWorld(px_horiz, py_horiz)
        players_world.append((x_cm, y_cm))

    if players_world:
        red_dots_world.append((frame_idx, video_time, players_world))

print(f"Converted {len(red_dots_world)} frames with player detections")
total_red_dots = sum(len(p) for _, _, p in red_dots_world)
print(f"Total player detections: {total_red_dots}")

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

# Pre-sort UWB positions
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
# ANALYSIS FUNCTIONS
# ============================================================================

def get_uwb_positions_at_time(target_seconds, offset):
    """Get all UWB tag positions at a specific time with offset applied"""
    target_dt = uwb_start + timedelta(seconds=target_seconds + offset)

    uwb_positions = {}
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
                uwb_positions[tag_id] = (x, y)

    return uwb_positions

def analyze_offset(offset_seconds):
    """Analyze sync quality for a given offset"""
    print(f"\n{'='*70}")
    print(f"Analyzing offset: {offset_seconds}s ({offset_seconds/60:.2f} min)")
    print(f"{'='*70}")

    # Sample frames for faster analysis (every 30 frames = ~1 second)
    sample_frames = red_dots_world[::30]
    print(f"Analyzing {len(sample_frames)} sample frames...")

    stats = {
        'offset_seconds': offset_seconds,
        'offset_minutes': offset_seconds / 60,
        'total_frames': len(sample_frames),
        'total_red_dots': 0,
        'total_blue_dots': 0,
        'tagged_players': 0,  # Red dots within 200cm of blue dot (would turn green)
        'untagged_players': 0,  # Red dots not within 200cm (stay red)
        'distances': [],  # All minimum distances
        'tagged_distances': [],  # Distances for tagged players
        'frames_with_tags': 0,
        'frames_with_players': 0,
        'frames_with_both': 0,
        'distance_histogram': defaultdict(int),  # Binned distances
    }

    for frame_idx, video_time, red_positions in sample_frames:
        stats['total_red_dots'] += len(red_positions)
        stats['frames_with_players'] += 1

        # Get UWB positions for this frame
        uwb_positions = get_uwb_positions_at_time(video_time, offset_seconds)

        if uwb_positions:
            stats['frames_with_tags'] += 1
            stats['total_blue_dots'] += len(uwb_positions)

        if red_positions and uwb_positions:
            stats['frames_with_both'] += 1

        # Calculate distances for each red dot
        for rx, ry in red_positions:
            if not uwb_positions:
                stats['untagged_players'] += 1
                continue

            # Find nearest UWB tag
            min_dist = float('inf')
            for tag_id, (ux, uy) in uwb_positions.items():
                dist = np.sqrt((rx - ux)**2 + (ry - uy)**2)
                min_dist = min(min_dist, dist)

            stats['distances'].append(min_dist)

            # Check if within tagging radius
            if min_dist <= TAGGING_RADIUS_CM:
                stats['tagged_players'] += 1
                stats['tagged_distances'].append(min_dist)
            else:
                stats['untagged_players'] += 1

            # Histogram (50cm bins)
            bin_idx = int(min_dist / 50) * 50
            stats['distance_histogram'][bin_idx] += 1

    # Calculate summary statistics
    if stats['distances']:
        stats['avg_distance'] = np.mean(stats['distances'])
        stats['median_distance'] = np.median(stats['distances'])
        stats['min_distance'] = np.min(stats['distances'])
        stats['max_distance'] = np.max(stats['distances'])
        stats['std_distance'] = np.std(stats['distances'])
    else:
        stats['avg_distance'] = float('inf')
        stats['median_distance'] = float('inf')
        stats['min_distance'] = float('inf')
        stats['max_distance'] = float('inf')
        stats['std_distance'] = 0

    if stats['total_red_dots'] > 0:
        stats['tagging_rate'] = stats['tagged_players'] / stats['total_red_dots']
    else:
        stats['tagging_rate'] = 0

    return stats

# ============================================================================
# RUN ANALYSIS
# ============================================================================

all_results = []

for offset in OFFSETS_TO_TEST:
    result = analyze_offset(offset)
    all_results.append(result)

# ============================================================================
# GENERATE REPORT
# ============================================================================

print(f"\n{'='*70}")
print(f"FULL GAME SYNC ANALYSIS REPORT")
print(f"{'='*70}\n")

print(f"Dataset: {len(red_dots_world)} frames, {total_red_dots} player detections")
print(f"Tagging radius: {TAGGING_RADIUS_CM}cm")
print(f"\n{'='*70}")
print(f"RESULTS BY OFFSET")
print(f"{'='*70}\n")

for result in all_results:
    print(f"Offset: {result['offset_seconds']}s ({result['offset_minutes']:.2f} min)")
    print(f"{'-'*70}")
    print(f"  Frame coverage:")
    print(f"    Frames with players: {result['frames_with_players']}")
    print(f"    Frames with UWB tags: {result['frames_with_tags']}")
    print(f"    Frames with both: {result['frames_with_both']}")
    print(f"\n  Player counts:")
    print(f"    Total red dots: {result['total_red_dots']}")
    print(f"    Tagged (GREEN - within {TAGGING_RADIUS_CM}cm): {result['tagged_players']} ({result['tagging_rate']*100:.1f}%)")
    print(f"    Untagged (RED - beyond {TAGGING_RADIUS_CM}cm): {result['untagged_players']} ({(1-result['tagging_rate'])*100:.1f}%)")
    print(f"\n  Distance statistics (cm):")
    print(f"    Average: {result['avg_distance']:.1f}cm ({result['avg_distance']/100:.2f}m)")
    print(f"    Median: {result['median_distance']:.1f}cm ({result['median_distance']/100:.2f}m)")
    print(f"    Min: {result['min_distance']:.1f}cm")
    print(f"    Max: {result['max_distance']:.1f}cm")
    print(f"    Std dev: {result['std_distance']:.1f}cm")

    if result['tagged_distances']:
        avg_tagged = np.mean(result['tagged_distances'])
        print(f"    Avg for tagged players: {avg_tagged:.1f}cm")

    print(f"\n  Distance histogram:")
    sorted_bins = sorted(result['distance_histogram'].items())
    for bin_start, count in sorted_bins[:10]:  # Show first 10 bins
        bin_end = bin_start + 50
        pct = (count / result['total_red_dots']) * 100 if result['total_red_dots'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {bin_start:4d}-{bin_end:4d}cm: {count:5d} players ({pct:5.1f}%) {bar}")

    print(f"\n")

# Find best offset
best_by_tagging = max(all_results, key=lambda x: x['tagging_rate'])
best_by_distance = min(all_results, key=lambda x: x['avg_distance'])

print(f"{'='*70}")
print(f"RECOMMENDATIONS")
print(f"{'='*70}\n")

print(f"Best by tagging rate:")
print(f"  Offset: {best_by_tagging['offset_seconds']}s ({best_by_tagging['offset_minutes']:.2f} min)")
print(f"  Tagging rate: {best_by_tagging['tagging_rate']*100:.1f}%")
print(f"  Avg distance: {best_by_tagging['avg_distance']:.1f}cm\n")

print(f"Best by average distance:")
print(f"  Offset: {best_by_distance['offset_seconds']}s ({best_by_distance['offset_minutes']:.2f} min)")
print(f"  Avg distance: {best_by_distance['avg_distance']:.1f}cm")
print(f"  Tagging rate: {best_by_distance['tagging_rate']*100:.1f}%\n")

if best_by_tagging['offset_seconds'] == best_by_distance['offset_seconds']:
    print(f"✅ CONSENSUS: Use offset {best_by_tagging['offset_seconds']}s ({best_by_tagging['offset_minutes']:.2f} min)")
    print(f"   This offset is best for both tagging rate AND distance accuracy!")
else:
    print(f"⚠️  TRADE-OFF DECISION NEEDED:")
    print(f"   - For maximum green tags: Use {best_by_tagging['offset_seconds']}s")
    print(f"   - For minimum distance error: Use {best_by_distance['offset_seconds']}s")

# Save detailed results
output_file = "full_game_sync_analysis.json"
with open(output_file, 'w') as f:
    # Convert numpy types to native Python for JSON serialization
    for result in all_results:
        for key, value in result.items():
            if isinstance(value, np.floating):
                result[key] = float(value)
            elif isinstance(value, np.integer):
                result[key] = int(value)
            elif isinstance(value, list) and value and isinstance(value[0], np.floating):
                result[key] = [float(v) for v in value]

    json.dump({
        'analysis_date': datetime.now().isoformat(),
        'dataset': {
            'red_coords_file': RED_COORDS_JSON,
            'total_frames': len(red_dots_world),
            'total_red_dots': total_red_dots,
            'uwb_tags': len(tag_data),
        },
        'parameters': {
            'tagging_radius_cm': TAGGING_RADIUS_CM,
            'fps': FPS,
        },
        'results': all_results,
        'best_by_tagging': {
            'offset_seconds': best_by_tagging['offset_seconds'],
            'tagging_rate': float(best_by_tagging['tagging_rate']),
        },
        'best_by_distance': {
            'offset_seconds': best_by_distance['offset_seconds'],
            'avg_distance': float(best_by_distance['avg_distance']),
        }
    }, f, indent=2)

print(f"\n✅ Detailed results saved to: {output_file}")
