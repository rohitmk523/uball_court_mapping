#!/usr/bin/env python3
"""
Visual Validation Tool - 3-Panel Stitcher

Creates a validation video with:
- Top: Original video with YOLO detection boxes
- Bottom-left: Red dots (from generate_red_dots.py)
- Bottom-right: Blue dots (from generate_blue_dots.py with sync offset)

This uses the ACTUAL red/blue dot generation scripts, so any fixes made
to those scripts will automatically be reflected here.

Usage:
    python visual_validation_stitch.py 00:15:00 00:15:10

This will generate a 10-second validation video starting at 15 minutes.
"""

import sys
import cv2
import numpy as np
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from app.services.player_detector import PlayerDetector

# Configuration
VIDEO_PATH = "GX020018_1080p.MP4"
SYNC_ANALYSIS_FILE = "cluster_sync_analysis.json"
FPS = 29.97

# Layout dimensions - Side-by-side layout
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

# Court videos: 4261x7341 (vertical), scale to match video height (1080)
# Maintain aspect ratio: width = 4261 * (1080/7341) = 627
COURT_DISPLAY_HEIGHT = ORIGINAL_HEIGHT  # 1080
COURT_DISPLAY_WIDTH = int(4261 * (ORIGINAL_HEIGHT / 7341))  # 627

# Total dimensions: all 3 side-by-side
TOTAL_WIDTH = ORIGINAL_WIDTH + (2 * COURT_DISPLAY_WIDTH)  # 1920 + 627 + 627 = 3174
TOTAL_HEIGHT = ORIGINAL_HEIGHT  # 1080


def time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_annotated_video(start_time, end_time, output_path, sync_offset):
    """Generate video with YOLO detection boxes + Tag IDs on original footage."""
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)

    print(f"\n1. Generating annotated video (YOLO11 boxes + Tag IDs on original)...")
    print(f"   Input: {VIDEO_PATH}")
    print(f"   Time: {start_time} to {end_time}")
    print(f"   Sync offset: {sync_offset}s for UWB association")

    # Load player detector with ByteTrack tracking
    player_detector = PlayerDetector(
        model_name="yolo11n.pt",
        confidence_threshold=0.3,
        enable_tracking=True,
        track_buffer=50
    )

    # Load calibration for court projection
    from app.services.calibration_integration import CalibrationIntegration
    calibration = CalibrationIntegration("data/calibration/1080p/homography.json")

    # Load UWB data and setup association
    from datetime import datetime, timedelta
    from bisect import bisect_left
    import json
    from pathlib import Path

    # Load UWB tags
    tags_dir = Path("data/tags")
    tag_data = {}
    for tag_file in tags_dir.glob('*.json'):
        with open(tag_file, 'r') as f:
            data = json.load(f)
            tag_data[data['tag_id']] = data['positions']

    # Get UWB start time
    first_tag_id = list(tag_data.keys())[0]
    uwb_start = datetime.fromisoformat(tag_data[first_tag_id][0]['datetime'])

    # Pre-sort UWB data
    sorted_tag_data = {}
    for tag_id, positions in tag_data.items():
        sorted_positions = []
        for pos in positions:
            dt = datetime.fromisoformat(pos['datetime'])
            sorted_positions.append((dt, pos['x'], pos['y']))
        sorted_positions.sort(key=lambda x: x[0])
        sorted_tag_data[tag_id] = sorted_positions

    print(f"   Loaded {len(tag_data)} UWB tags for association")

    # Association log
    association_log = {}

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    num_frames = end_frame - start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))

    print(f"   Processing {num_frames} frames...")

    # Helper function to get UWB positions at a specific time
    def get_uwb_positions_at_time(target_dt):
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

    for frame_idx in tqdm(range(num_frames), desc="   Annotating"):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current video time
        current_frame = start_frame + frame_idx
        video_time_sec = current_frame / fps

        # Get UWB positions at synchronized time
        uwb_time_sec = video_time_sec + sync_offset
        target_dt = uwb_start + timedelta(seconds=uwb_time_sec)
        uwb_positions = get_uwb_positions_at_time(target_dt)

        # Run YOLO detection with tracking
        tracked_players = player_detector.track_players(frame)

        # Draw bounding boxes with Tag IDs
        for player in tracked_players:
            bbox = player['bbox']
            conf = player['confidence']
            track_id = player.get('track_id')
            x1, y1, x2, y2 = map(int, bbox)

            # Project to court to find UWB association
            bx, by = player['bottom']
            tag_id = None
            try:
                court_x, court_y = calibration.image_to_court(bx, by)

                # Find nearest UWB tag
                min_dist = float('inf')
                nearest_tag = None
                
                # UWB coordinates need to be rotated to match calibration system
                # calibration.image_to_court returns (x_cal, y_cal)
                # uwb_positions contains (x_uwb, y_uwb)
                
                for uwb_tag_id, (uwb_x, uwb_y) in uwb_positions.items():
                    # Rotate UWB point to match calibration coordinate system
                    cal_uwb_x, cal_uwb_y = calibration.rotate_uwb_to_calibration(uwb_x, uwb_y)
                    
                    # Calculate distance in calibration coordinate system
                    dist = np.sqrt((court_x - cal_uwb_x)**2 + (court_y - cal_uwb_y)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_tag = uwb_tag_id

                if min_dist < 300:  # Increased threshold to 300cm (approx 3m)
                    tag_id = nearest_tag
                    player['tag_id'] = tag_id  # Attach to player dict
                    # Log association
                    if track_id not in association_log:
                        association_log[track_id] = []
                    association_log[track_id].append({
                        'frame': frame_idx,
                        'tag_id': tag_id,
                        'distance': float(min_dist)
                    })
            except Exception as e:
                # print(f"Error in association: {e}")
                pass

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label: Tag ID if found, otherwise Track ID
            if tag_id is not None:
                label = f"Tag {tag_id}"
                color = (0, 255, 0)  # Green for tagged
            else:
                label = f"Track {track_id}" if track_id is not None else "Track ?"
                color = (0, 165, 255)  # Orange for untagged

            cv2.putText(frame, label,
                      (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add info overlay
        current_time = start_seconds + (frame_idx / fps)
        tagged_count = sum(1 for p in tracked_players if p.get('tag_id') is not None)
        info_text = f"Time: {int(current_time//60):02d}:{int(current_time%60):02d}.{int((current_time%1)*10)} | Players: {len(tracked_players)} | Tagged: {tagged_count}"

        # Draw background for text
        (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (5, 5), (text_width + 15, text_height + 20), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Save association log
    log_path = output_path.replace('.mp4', '_associations.json')
    with open(log_path, 'w') as f:
        json.dump(association_log, f, indent=2)

    print(f"   ✅ Annotated video: {output_path}")
    print(f"   ✅ Association log: {log_path}")
    print(f"   Track→Tag associations: {len(association_log)} unique tracks")


def generate_red_dots(start_time, end_time, output_path, video_path):
    """Call generate_red_dots.py script."""
    print(f"\n2. Generating RED dots video...")
    print(f"   Video: {video_path}")
    print(f"   Time window: {start_time} to {end_time} (VIDEO TIME)")

    cmd = [
        'python', 'generate_red_dots.py',
        '--start', start_time,
        '--end', end_time,
        '--resolution', 'highres',
        '--video', video_path,  # Pass the video path
        '--output', output_path
    ]

    # Run without capturing output to show real-time progress
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"   ❌ ERROR generating red dots")
        sys.exit(1)

    print(f"   ✅ Red dots video: {output_path}")


def generate_blue_dots(start_time, end_time, sync_offset, output_path):
    """Call generate_blue_dots.py script with sync offset applied."""
    # Convert video time to UWB time
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)

    # Apply sync offset: video time + offset = UWB time
    uwb_start_seconds = start_seconds + sync_offset
    uwb_end_seconds = end_seconds + sync_offset

    uwb_start_time = seconds_to_time(uwb_start_seconds)
    uwb_end_time = seconds_to_time(uwb_end_seconds)

    print(f"\n3. Generating BLUE dots video...")
    print(f"   Video time: {start_time} to {end_time}")
    print(f"   Sync offset: +{sync_offset}s ({sync_offset/60:.1f} min)")
    print(f"   UWB time: {uwb_start_time} to {uwb_end_time}")

    cmd = [
        'python', 'generate_blue_dots.py',
        '--start', uwb_start_time,
        '--end', uwb_end_time,
        '--resolution', 'highres',
        '--output', output_path
    ]

    # Run without capturing output to show real-time progress
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"   ❌ ERROR generating blue dots")
        sys.exit(1)

    print(f"   ✅ Blue dots video: {output_path}")


def stitch_videos(annotated_path, red_dots_path, blue_dots_path, output_path, sync_offset):
    """Stitch three videos into final layout (side-by-side)."""
    print(f"\n4. Stitching videos together (side-by-side layout)...")

    # Open all three videos
    cap_annotated = cv2.VideoCapture(annotated_path)
    cap_red = cv2.VideoCapture(red_dots_path)
    cap_blue = cv2.VideoCapture(blue_dots_path)

    fps = cap_annotated.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap_annotated.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   Layout: {TOTAL_WIDTH}x{TOTAL_HEIGHT} (side-by-side)")
    print(f"   Left: YOLO11 video ({ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT})")
    print(f"   Middle: Red dots ({COURT_DISPLAY_WIDTH}x{COURT_DISPLAY_HEIGHT})")
    print(f"   Right: Blue dots ({COURT_DISPLAY_WIDTH}x{COURT_DISPLAY_HEIGHT})")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TOTAL_WIDTH, TOTAL_HEIGHT))

    print(f"   Processing {num_frames} frames...")

    for frame_idx in tqdm(range(num_frames), desc="   Stitching"):
        # Read frames
        ret1, frame_annotated = cap_annotated.read()
        ret2, frame_red = cap_red.read()
        ret3, frame_blue = cap_blue.read()

        if not (ret1 and ret2 and ret3):
            break

        # Resize court videos to display size (maintain aspect ratio)
        frame_red_resized = cv2.resize(frame_red, (COURT_DISPLAY_WIDTH, COURT_DISPLAY_HEIGHT))
        frame_blue_resized = cv2.resize(frame_blue, (COURT_DISPLAY_WIDTH, COURT_DISPLAY_HEIGHT))

        # Create combined frame - side-by-side layout
        combined = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)

        # Left: YOLO11 annotated video
        combined[0:ORIGINAL_HEIGHT, 0:ORIGINAL_WIDTH] = frame_annotated

        # Middle: Red dots
        red_start_x = ORIGINAL_WIDTH
        red_end_x = ORIGINAL_WIDTH + COURT_DISPLAY_WIDTH
        combined[0:COURT_DISPLAY_HEIGHT, red_start_x:red_end_x] = frame_red_resized

        # Right: Blue dots
        blue_start_x = ORIGINAL_WIDTH + COURT_DISPLAY_WIDTH
        blue_end_x = ORIGINAL_WIDTH + (2 * COURT_DISPLAY_WIDTH)
        combined[0:COURT_DISPLAY_HEIGHT, blue_start_x:blue_end_x] = frame_blue_resized

        # Add labels at the top of each panel
        # YOLO video label
        cv2.rectangle(combined, (10, 10), (550, 60), (0, 0, 0), -1)
        cv2.putText(combined, "YOLO11 + Tag IDs",
                   (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Red dots label
        red_label_x = red_start_x + 10
        cv2.rectangle(combined, (red_label_x, 10),
                     (red_label_x + 400, 60), (0, 0, 0), -1)
        cv2.putText(combined, "RED: YOLO->Court",
                   (red_label_x + 5, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Blue dots label with sync info
        blue_label_x = blue_start_x + 10
        cv2.rectangle(combined, (blue_label_x, 10),
                     (blue_label_x + 550, 100), (0, 0, 0), -1)
        cv2.putText(combined, "BLUE: UWB Tags",
                   (blue_label_x + 5, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(combined, f"Sync: +{sync_offset}s ({sync_offset/60:.1f}min)",
                   (blue_label_x + 5, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(combined)

    # Cleanup
    cap_annotated.release()
    cap_red.release()
    cap_blue.release()
    out.release()

    print(f"   ✅ Stitched video: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python visual_validation_stitch.py <start_time> <end_time> [sync_offset] [video_path]")
        print("Example: python visual_validation_stitch.py 00:15:00 00:15:10")
        print("Example: python visual_validation_stitch.py 00:15:00 00:15:10 1194")
        print("Example: python visual_validation_stitch.py 00:15:00 00:15:10 8 GX010018_1080p.MP4")
        sys.exit(1)

    start_time = sys.argv[1]
    end_time = sys.argv[2]

    # Get video path from command line or use default
    global VIDEO_PATH
    if len(sys.argv) >= 5:
        VIDEO_PATH = sys.argv[4]
    elif len(sys.argv) >= 4 and sys.argv[3].endswith('.MP4'):
        # If 3rd arg is a video file, use it as video path
        VIDEO_PATH = sys.argv[3]

    # Load sync offset
    print("=" * 70)
    print("Visual Validation Tool - 3-Panel Stitcher")
    print("=" * 70)
    print(f"Video: {VIDEO_PATH}")
    print(f"Time window: {start_time} to {end_time}")

    # Try to load sync offset from file or command line
    if len(sys.argv) >= 4 and not sys.argv[3].endswith('.MP4'):
        sync_offset = float(sys.argv[3])
        print(f"Using sync offset from command line: {sync_offset}s")
    else:
        try:
            with open(SYNC_ANALYSIS_FILE) as f:
                sync_data = json.load(f)
                sync_offset = sync_data['best_offset_seconds']
            print(f"Loaded sync offset from {SYNC_ANALYSIS_FILE}: {sync_offset}s")
        except:
            # Default from cluster analysis
            sync_offset = 1194.0
            print(f"Using default sync offset: {sync_offset}s")

    print(f"Sync offset: {sync_offset}s ({sync_offset/60:.1f} min)")
    print("=" * 70)

    # Generate temporary video files
    start_label = start_time.replace(':', '')
    end_label = end_time.replace(':', '')

    annotated_path = f"temp_annotated_{start_label}_{end_label}.mp4"
    red_dots_path = f"temp_red_{start_label}_{end_label}.mp4"
    blue_dots_path = f"temp_blue_{start_label}_{end_label}.mp4"
    output_path = f"validation_{start_label}_to_{end_label}.mp4"

    try:
        # Generate all three videos
        generate_annotated_video(start_time, end_time, annotated_path, sync_offset)
        generate_red_dots(start_time, end_time, red_dots_path, VIDEO_PATH)
        generate_blue_dots(start_time, end_time, sync_offset, blue_dots_path)

        # Stitch them together
        stitch_videos(annotated_path, red_dots_path, blue_dots_path, output_path, sync_offset)

        print("\n" + "=" * 70)
        print(f"✅ VALIDATION VIDEO COMPLETE!")
        print(f"   Output: {output_path}")
        print(f"   Duration: {time_to_seconds(end_time) - time_to_seconds(start_time)} seconds")
        print(f"   Resolution: {TOTAL_WIDTH}x{TOTAL_HEIGHT} (side-by-side)")
        print("\nLayout (Side-by-Side):")
        print("  ┌──────────────┬───────────┬───────────┐")
        print("  │   YOLO11     │    RED    │   BLUE    │")
        print("  │  Detections  │   Dots    │   Dots    │")
        print("  │  (Original)  │ (YOLO->   │ (UWB +    │")
        print("  │              │  Court)   │  sync)    │")
        print("  │ 1920x1080    │ 627x1080  │ 627x1080  │")
        print("  └──────────────┴───────────┴───────────┘")
        print(f"  Total: {TOTAL_WIDTH}x{TOTAL_HEIGHT} (3174x1080)")
        print("=" * 70)

    finally:
        # Cleanup temporary files
        print("\n5. Cleaning up temporary files...")
        for temp_file in [annotated_path, red_dots_path, blue_dots_path]:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
                print(f"   Deleted: {temp_file}")


if __name__ == "__main__":
    main()
