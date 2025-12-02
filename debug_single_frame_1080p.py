#!/usr/bin/env python3
"""
Debug script to test court visualization on a single frame using 1080p calibration.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.player_detector import PlayerDetector
from app.services.calibration_integration import CalibrationIntegration
from app.core.court_config import CourtVideoConfig
from app.services.court_video_generator import CourtVideoGenerator

def debug_single_frame(frame_number=100, video_path="GX020018_1080p.MP4"):
    """Debug processing of a single frame using 1080p calibration."""

    print("=" * 70)
    print("DEBUG: Single Frame Court Visualization (1080p)")
    print("=" * 70)

    # Load video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print(f"ERROR: Could not read frame {frame_number}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_time = frame_number / fps

    print(f"\nFrame: {frame_number}")
    print(f"Time: {video_time:.2f}s")
    print(f"Video resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Initialize components with 1080p calibration
    print("\n1. Initializing components...")
    config = CourtVideoConfig(
        video_path=video_path,
        calibration_file="data/calibration/1080p/homography.json",
        court_image_file="data/calibration/court_image.png",
        tags_dir="data/tags",
        log_file="data/logs/session_*.log",
        sync_offset_seconds=49.422,
        yolo_model="yolov8n.pt",
        yolo_confidence=0.5,
        yolo_device="mps"
    )

    generator = CourtVideoGenerator(config)
    # Manually load the court canvas (normally done in process_video())
    generator.court_canvas_template = generator.load_court_canvas()
    print(f"   Court canvas: {generator.canvas_width}x{generator.canvas_height}")
    print(f"   Scale: X={generator.scale_x:.3f}, Y={generator.scale_y:.3f} px/cm")

    # Detect players
    print("\n2. Detecting players with YOLO...")
    detector = PlayerDetector("yolov8n.pt", 0.5, "mps")
    detections = detector.detect_players(frame)
    print(f"   Detected {len(detections)} players")

    for i, det in enumerate(detections):
        print(f"   Player {i}: bbox={det['bbox']}, bottom_center={det['bottom']}, conf={det['confidence']:.2f}")

    # Project players
    print("\n3. Projecting players to court coordinates...")
    players = []
    for idx, det in enumerate(detections):
        det['index'] = idx
        player_pos = generator.project_player_to_court(det)

        if player_pos:
            players.append(player_pos)
            print(f"   Player {idx}: Video {det['bottom']} -> Court ({player_pos.court_x:.1f}, {player_pos.court_y:.1f}) -> Canvas ({player_pos.canvas_x}, {player_pos.canvas_y})")
            print(f"      Video bottom: {det['bottom']}")
            print(f"      Court cm: ({player_pos.court_x:.1f}, {player_pos.court_y:.1f})")
            print(f"      Canvas px: ({player_pos.canvas_x}, {player_pos.canvas_y})")
        else:
            bx, by = det['bottom']
            cx, cy = generator.calibration.image_to_court(bx, by)
            print(f"   Player {idx}: Video {det['bottom']} -> Court ({cx:.1f}, {cy:.1f}) [OUT OF BOUNDS]")

    # Create visualization
    print("\n4. Creating visualization...")
    canvas = generator.court_canvas_template.copy()

    # Draw coordinate system for reference
    # Draw origin marker
    cv2.circle(canvas, (0, canvas.shape[0]), 30, (0, 0, 255), -1)  # Red dot at origin
    cv2.putText(canvas, "ORIGIN", (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Draw center marker
    center_x = canvas.shape[1] // 2
    center_y = canvas.shape[0] // 2
    cv2.circle(canvas, (center_x, center_y), 30, (255, 0, 255), -1)  # Magenta at center
    cv2.putText(canvas, "CENTER", (center_x + 40, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    # Draw players (LARGE for debugging)
    print(f"\n5. Drawing {len(players)} players...")
    for player in players:
        # Check if position is valid
        if 0 <= player.canvas_x < canvas.shape[1] and 0 <= player.canvas_y < canvas.shape[0]:
            # Draw VERY LARGE yellow circle for visibility
            cv2.circle(canvas, (player.canvas_x, player.canvas_y), 100, (0, 255, 255), -1)  # Yellow, filled, 100px
            cv2.circle(canvas, (player.canvas_x, player.canvas_y), 100, (0, 0, 0), 8)  # Thick black outline
            # Draw player index
            cv2.putText(canvas, f"P{player.detection_index}", (player.canvas_x - 50, player.canvas_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            print(f"   Drew player {player.detection_index} at ({player.canvas_x}, {player.canvas_y})")
        else:
            print(f"   Player {player.detection_index} at ({player.canvas_x}, {player.canvas_y}) is OUTSIDE canvas!")

    # Save outputs
    print("\n6. Saving outputs...")

    # Save video frame with detection boxes
    frame_with_boxes = frame.copy()
    for det in detections:
        bbox = det['bbox']
        bottom = det['bottom']
        # Draw bounding box
        cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        # Draw bottom center point
        cv2.circle(frame_with_boxes, bottom, 10, (0, 0, 255), -1)
        # Draw player index
        cv2.putText(frame_with_boxes, f"P{detections.index(det)}", (bbox[0], bbox[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("debug_video_frame_1080p.jpg", frame_with_boxes)
    print(f"   Saved video frame: debug_video_frame_1080p.jpg")

    # Save court visualization
    cv2.imwrite("debug_court_output_1080p.jpg", canvas)
    print(f"   Saved court output: debug_court_output_1080p.jpg")

    # Create side-by-side comparison
    # Resize video frame to match court height
    video_height = canvas.shape[0]
    video_scale = video_height / frame_with_boxes.shape[0]
    video_resized = cv2.resize(frame_with_boxes,
                               (int(frame_with_boxes.shape[1] * video_scale), video_height))

    # Create side-by-side image
    comparison = np.hstack([video_resized, canvas])

    # Add labels
    cv2.putText(comparison, "VIDEO FRAME (1080p)", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    cv2.putText(comparison, "COURT PROJECTION", (video_resized.shape[1] + 50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5)

    cv2.imwrite("debug_comparison_1080p.jpg", comparison)
    print(f"   Saved comparison: debug_comparison_1080p.jpg")

    # Save smaller version for easy viewing
    scale = 0.2
    small = cv2.resize(comparison, (int(comparison.shape[1] * scale), int(comparison.shape[0] * scale)))
    cv2.imwrite("debug_comparison_small_1080p.jpg", small)
    print(f"   Saved small comparison: debug_comparison_small_1080p.jpg")

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE!")
    print("=" * 70)
    print("\nCheck the output files:")
    print("  - debug_video_frame_1080p.jpg (video frame with detection boxes)")
    print("  - debug_court_output_1080p.jpg (full court visualization)")
    print("  - debug_comparison_1080p.jpg (side-by-side video + court)")
    print("  - debug_comparison_small_1080p.jpg (20% scaled for easier viewing)")
    print("\nLook for:")
    print("  - VIDEO: Green boxes = detections, Red dots = bottom-center points")
    print("  - COURT: YELLOW dots = Players")
    print("  - Check if players in video appear on court!")


if __name__ == "__main__":
    frame_num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    debug_single_frame(frame_num)
