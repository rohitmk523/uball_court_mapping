#!/usr/bin/env python3
"""
Generate video with Tag ID labels above bounding boxes.
Shows "Tag XXXXX" if UWB tag found within 200cm, otherwise "Track YY" as temporary ID.
Outputs both MP4 video and JSON log of Track→Tag associations for reprocessing.

Usage:
    python generate_video_with_tags.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:25:00 --sync-offset 8 --output tagged_video.mp4
"""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from bisect import bisect_left
from tqdm import tqdm

from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration


def parse_time(time_str):
    """Convert HH:MM:SS to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def time_to_str(seconds):
    """Convert seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_uwb_data(tags_dir):
    """Load and sort UWB tag data."""
    tags_dir = Path(tags_dir)
    tag_data = {}

    for tag_file in tags_dir.glob('*.json'):
        with open(tag_file, 'r') as f:
            data = json.load(f)
            tag_data[data['tag_id']] = data['positions']

    # Get UWB start time from first tag
    first_tag_id = list(tag_data.keys())[0]
    uwb_start = datetime.fromisoformat(tag_data[first_tag_id][0]['datetime'])

    # Pre-sort UWB data for binary search
    sorted_tag_data = {}
    for tag_id, positions in tag_data.items():
        sorted_positions = []
        for pos in positions:
            dt = datetime.fromisoformat(pos['datetime'])
            sorted_positions.append((dt, pos['x'], pos['y']))
        sorted_positions.sort(key=lambda x: x[0])
        sorted_tag_data[tag_id] = sorted_positions

    return sorted_tag_data, uwb_start


def get_uwb_positions_at_time(target_dt, sorted_tag_data, time_tolerance=5.0):
    """Get UWB positions at a specific datetime with binary search."""
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
            if time_diff < time_tolerance:
                dt, x, y = sorted_positions[best_idx]
                uwb_positions[tag_id] = (x, y)

    return uwb_positions


def find_nearest_tag(court_x, court_y, uwb_positions, threshold=200):
    """Find nearest UWB tag to court position within threshold (pixels)."""
    min_dist = float('inf')
    nearest_tag = None

    for tag_id, (uwb_x, uwb_y) in uwb_positions.items():
        dist = np.sqrt((court_x - uwb_x)**2 + (court_y - uwb_y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_tag = tag_id

    if min_dist < threshold:
        return nearest_tag, min_dist
    return None, None


def generate_tagged_video(video_path, start_time, end_time, sync_offset, output_path,
                         json_output_path=None, proximity_threshold=200):
    """
    Generate video with Tag ID labels.

    Args:
        video_path: Path to input video
        start_time: Start time in HH:MM:SS format
        end_time: End time in HH:MM:SS format
        sync_offset: Sync offset in seconds (UWB time = video time + offset)
        output_path: Path to output MP4 file
        json_output_path: Optional path to JSON log file
        proximity_threshold: Distance threshold in pixels (default 200)
    """
    print(f"\n{'='*70}")
    print(f"Tag ID Video Generator")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Time: {start_time} → {end_time}")
    print(f"Sync offset: {sync_offset}s")
    print(f"Proximity threshold: {proximity_threshold} pixels")
    print(f"Output video: {output_path}")
    if json_output_path:
        print(f"Output log: {json_output_path}")
    print(f"{'='*70}\n")

    # Parse times
    start_seconds = parse_time(start_time)
    end_seconds = parse_time(end_time)

    # Load components
    print("Loading components...")
    player_detector = PlayerDetector(
        model_name="yolo11n.pt",
        confidence_threshold=0.3,
        enable_tracking=True,
        track_buffer=50
    )
    calibration = CalibrationIntegration("data/calibration/1080p/homography.json")
    sorted_tag_data, uwb_start = load_uwb_data("data/tags")
    print(f"✓ Loaded {len(sorted_tag_data)} UWB tags\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    num_frames = end_frame - start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Association tracking
    association_log = {}
    frame_data = []

    print(f"Processing {num_frames} frames ({num_frames/fps:.1f}s @ {fps:.2f} fps)...\n")

    for frame_idx in tqdm(range(num_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current video time
        current_frame = start_frame + frame_idx
        video_time_sec = current_frame / fps

        # Get UWB positions at synchronized time
        uwb_time_sec = video_time_sec + sync_offset
        target_dt = uwb_start + timedelta(seconds=uwb_time_sec)
        uwb_positions = get_uwb_positions_at_time(target_dt, sorted_tag_data)

        # Run YOLO detection with tracking
        tracked_players = player_detector.track_players(frame)

        # Store frame data
        frame_info = {
            'frame': frame_idx,
            'time': video_time_sec,
            'players': []
        }

        # Process each player
        for player in tracked_players:
            bbox = player['bbox']
            conf = player['confidence']
            track_id = player.get('track_id')
            x1, y1, x2, y2 = bbox

            # Project to court to find UWB association
            bx, by = player['bottom']
            tag_id = None
            distance = None

            try:
                court_x, court_y = calibration.image_to_court(bx, by)
                tag_id, distance = find_nearest_tag(court_x, court_y, uwb_positions, proximity_threshold)

                if tag_id is not None:
                    # Log association
                    if track_id not in association_log:
                        association_log[track_id] = []
                    association_log[track_id].append({
                        'frame': frame_idx,
                        'time': video_time_sec,
                        'tag_id': tag_id,
                        'distance': float(distance),
                        'court_position': [float(court_x), float(court_y)]
                    })
            except:
                pass

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label: Tag ID if found, otherwise Track ID
            if tag_id is not None:
                label = f"Tag {tag_id}"
                color = (0, 255, 0)  # Green for tagged
                status = 'tagged'
            else:
                label = f"Track {track_id}" if track_id is not None else "Track ?"
                color = (0, 165, 255)  # Orange for untagged
                status = 'untagged'

            # Add background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Store player data
            frame_info['players'].append({
                'track_id': track_id,
                'tag_id': tag_id,
                'bbox': bbox,
                'confidence': conf,
                'status': status,
                'distance': float(distance) if distance is not None else None
            })

        # Add info overlay
        tagged_count = sum(1 for p in tracked_players if p.get('tag_id') is not None)
        info_text = f"Time: {time_to_str(video_time_sec)} | Players: {len(tracked_players)} | Tagged: {tagged_count}/{len(tracked_players)}"

        (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (5, 5), (text_width + 15, text_height + 20), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_data.append(frame_info)

    cap.release()
    out.release()

    # Save JSON log
    if json_output_path:
        log_data = {
            'video': video_path,
            'time_range': [start_time, end_time],
            'sync_offset': sync_offset,
            'proximity_threshold': proximity_threshold,
            'total_frames': num_frames,
            'associations': association_log,
            'frames': frame_data,
            'summary': {
                'unique_tracks': len(association_log),
                'total_associations': sum(len(v) for v in association_log.values())
            }
        }

        with open(json_output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"Output video: {output_path}")
    if json_output_path:
        print(f"Output log: {json_output_path}")
    print(f"\nStatistics:")
    print(f"  Frames processed: {num_frames}")
    print(f"  Duration: {num_frames/fps:.1f}s")
    print(f"  Unique tracks: {len(association_log)}")
    print(f"  Total Track→Tag associations: {sum(len(v) for v in association_log.values())}")
    print(f"\nTrack→Tag Mapping:")
    for track_id, associations in sorted(association_log.items()):
        tag_ids = set(a['tag_id'] for a in associations)
        if len(tag_ids) == 1:
            tag_id = list(tag_ids)[0]
            print(f"  Track {track_id:3d} → Tag {tag_id} ({len(associations)} frames)")
        else:
            print(f"  Track {track_id:3d} → Multiple tags: {tag_ids} (WARNING: ID switching!)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate video with Tag ID labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5-minute video with Tag IDs
  python generate_video_with_tags.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:25:00 --sync-offset 8 --output tagged.mp4

  # With JSON log for reprocessing
  python generate_video_with_tags.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:25:00 --sync-offset 8 --output tagged.mp4 --json tagged.json

  # Adjust proximity threshold
  python generate_video_with_tags.py --video GX010018_1080p.MP4 --start 00:20:00 --end 00:25:00 --sync-offset 8 --output tagged.mp4 --threshold 150
        """
    )

    parser.add_argument('--video', required=True, help='Input video path (e.g., GX010018_1080p.MP4)')
    parser.add_argument('--start', required=True, help='Start time (HH:MM:SS)')
    parser.add_argument('--end', required=True, help='End time (HH:MM:SS)')
    parser.add_argument('--sync-offset', type=int, required=True, help='Sync offset in seconds')
    parser.add_argument('--output', required=True, help='Output video path (e.g., tagged.mp4)')
    parser.add_argument('--json', dest='json_output', help='Optional JSON log output path')
    parser.add_argument('--threshold', type=int, default=200, help='Proximity threshold in pixels (default: 200)')

    args = parser.parse_args()

    generate_tagged_video(
        video_path=args.video,
        start_time=args.start,
        end_time=args.end,
        sync_offset=args.sync_offset,
        output_path=args.output,
        json_output_path=args.json_output,
        proximity_threshold=args.threshold
    )
