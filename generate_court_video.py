#!/usr/bin/env python3
"""
Generate court-centric video with UWB tags and player detections.
"""
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from app.services.dxf_parser import parse_court_dxf
from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration

# Configuration
VIDEO_PATH = "GX020018_1080p.MP4"
CALIBRATION_FILE = "data/calibration/1080p/homography.json"
TAGS_DIR = Path("data/tags")
OUTPUT_VIDEO = "output_court_video.mp4"

# Canvas size (will be rotated 90° CW)
HORIZONTAL_WIDTH = 1200
HORIZONTAL_HEIGHT = 800

# Sync offset
SYNC_OFFSET = 0.0  # No offset, start from actual timestamps

# Frame range
START_FRAME = 41360  # Where tags are spread
DURATION_MINUTES = 5
FPS = 29.97
NUM_FRAMES = int(DURATION_MINUTES * 60 * FPS)  # ~8991 frames

print(f"Court Video Generator")
print(f"=" * 70)
print(f"Video: {VIDEO_PATH}")
print(f"Start frame: {START_FRAME}")
print(f"Duration: {DURATION_MINUTES} minutes ({NUM_FRAMES} frames)")
print(f"Sync offset: {SYNC_OFFSET}s")
print(f"Output: {OUTPUT_VIDEO}")
print(f"=" * 70)

# Load court geometry
print("\nLoading court geometry...")
geometry = parse_court_dxf(Path('court_2.dxf'))
courtBounds = geometry.bounds

print(f"Court bounds: X({courtBounds.min_x:.1f} to {courtBounds.max_x:.1f}), Y({courtBounds.min_y:.1f} to {courtBounds.max_y:.1f})")

# Calculate scale (like tag_viewer)
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

# Initialize player detector and calibration
print("\nInitializing player detection...")
player_detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.3)
calibration = CalibrationIntegration(CALIBRATION_FILE)

# Open video
print("\nOpening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video {VIDEO_PATH}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames at {video_fps} fps")

# Set start position
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

# Create video writer (vertical output after rotation)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_fps, (HORIZONTAL_HEIGHT, HORIZONTAL_WIDTH))

print(f"\nProcessing frames {START_FRAME} to {START_FRAME + NUM_FRAMES}...")

for i in tqdm(range(NUM_FRAMES)):
    frame_number = START_FRAME + i
    video_time = frame_number / video_fps

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print(f"\nEnd of video at frame {frame_number}")
        break

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
        if circle[1] > 300:  # Skip large circles
            continue
        center = worldToScreen(*circle[0])
        radius = int(circle[1] * actualScale)
        cv2.circle(canvas, center, radius, (255, 255, 255), 1)

    # Get UWB tags at this time
    target_dt = uwb_start + timedelta(seconds=(video_time + SYNC_OFFSET))

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

            # Draw blue tag
            cv2.circle(canvas, (sx, sy), 8, (255, 100, 0), -1)  # Blue filled
            cv2.circle(canvas, (sx, sy), 8, (255, 255, 255), 1)  # White outline

    # Detect and project players (RED)
    detections = player_detector.detect_players(frame)

    for detection in detections:
        # Get bottom-center point
        bx, by = detection['bottom']

        try:
            # Project to canvas using homography
            canvas_x, canvas_y = calibration.image_to_court(bx, by)
            canvas_x, canvas_y = int(canvas_x), int(canvas_y)

            # The homography returns coordinates on VERTICAL canvas
            # We need to un-rotate to get horizontal canvas coordinates
            # Vertical (x, y) after 90° CW from Horizontal (X, Y):
            #   x = Y, y = W - X
            # So reverse: X = W - y, Y = x
            horiz_x = HORIZONTAL_WIDTH - canvas_y
            horiz_y = canvas_x

            # Check bounds
            if 0 <= horiz_x < HORIZONTAL_WIDTH and 0 <= horiz_y < HORIZONTAL_HEIGHT:
                # Draw red player
                cv2.circle(canvas, (horiz_x, horiz_y), 10, (0, 0, 255), -1)  # Red filled
                cv2.circle(canvas, (horiz_x, horiz_y), 10, (255, 255, 255), 2)  # White outline
        except:
            pass

    # Rotate 90° clockwise for vertical output
    canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

    # Write frame
    out.write(canvas_rotated)

# Cleanup
cap.release()
out.release()

print(f"\n✓ Video saved to: {OUTPUT_VIDEO}")
print(f"Duration: {NUM_FRAMES / video_fps / 60:.1f} minutes")
