#!/usr/bin/env python3
"""Verify 1080p calibration by visualizing the points and transformation"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path

# Get video path from command line or use default
video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("GX010018_1080p.MP4")

# Paths for 1080p
calib_dir = Path("data/calibration/1080p")
calib_file = calib_dir / "homography.json"
court_image_path = Path("data/calibration/court_image.png")

print("="*70)
print("VERIFYING 1080p CALIBRATION")
print("="*70)

# Load calibration data
with open(calib_file) as f:
    calib_data = json.load(f)

court_points = np.array(calib_data["court_points"], dtype=np.float32)
video_points = np.array(calib_data["video_points"], dtype=np.float32)
H = np.array(calib_data["homography"], dtype=np.float32)

print(f"\nLoaded calibration with {len(court_points)} point pairs")
print(f"Inliers: {calib_data.get('inliers', 'N/A')}")
print(f"Outliers: {calib_data.get('outliers', 'N/A')}")

# Load images
print("\nLoading images...")
court_img = cv2.imread(str(court_image_path))
# Rotate court image 90 degrees clockwise to match calibration tool
court_img = cv2.rotate(court_img, cv2.ROTATE_90_CLOCKWISE)
print("✓ Rotated court image 90 degrees clockwise")

# Extract frame 100 from 1080p video
cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, video_img = cap.read()
cap.release()
if not ret:
    raise ValueError("Could not read frame 100 from 1080p video")
print(f"✓ Loaded frame 100 from {video_path.name} ({video_img.shape[1]}x{video_img.shape[0]})")

# Create visualization 1: Side-by-side with points marked
print("\nCreating verification images...")
court_display = court_img.copy()
video_display = video_img.copy()

# Draw points on court
for i, pt in enumerate(court_points):
    pt = tuple(map(int, pt))
    cv2.circle(court_display, pt, 15, (0, 0, 255), -1)
    cv2.putText(court_display, str(i+1), (pt[0]+20, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

# Draw points on video
for i, pt in enumerate(video_points):
    pt = tuple(map(int, pt))
    cv2.circle(video_display, pt, 8, (0, 255, 0), -1)
    cv2.putText(video_display, str(i+1), (pt[0]+12, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Resize to same height for side-by-side
height = min(court_display.shape[0], video_display.shape[0])
court_aspect = court_display.shape[1] / court_display.shape[0]
video_aspect = video_display.shape[1] / video_display.shape[0]

court_width = int(height * court_aspect)
video_width = int(height * video_aspect)

court_resized = cv2.resize(court_display, (court_width, height))
video_resized = cv2.resize(video_display, (video_width, height))

# Create side-by-side
side_by_side = np.hstack([court_resized, video_resized])

# Add labels
cv2.putText(side_by_side, "COURT (Rotated 90°)", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.putText(side_by_side, "COURT (Rotated 90°)", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

cv2.putText(side_by_side, "VIDEO (1080p)", (court_width + 50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.putText(side_by_side, "VIDEO (1080p)", (court_width + 50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

points_output = calib_dir / "calibration_verification_points.jpg"
cv2.imwrite(str(points_output), side_by_side)
print(f"✓ Saved: {points_output}")

# Create visualization 2: Warped video overlaid on court
video_warped = cv2.warpPerspective(video_img, H,
                                   (court_img.shape[1], court_img.shape[0]),
                                   flags=cv2.WARP_INVERSE_MAP)

# Create semi-transparent overlay
alpha = 0.5
overlay = cv2.addWeighted(court_img, alpha, video_warped, 1-alpha, 0)

# Mark the points on overlay
for i, pt in enumerate(court_points):
    pt = tuple(map(int, pt))
    cv2.circle(overlay, pt, 12, (255, 0, 255), -1)
    cv2.putText(overlay, str(i+1), (pt[0]+15, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

overlay_output = calib_dir / "calibration_verification_overlay.jpg"
cv2.imwrite(str(overlay_output), overlay)
print(f"✓ Saved: {overlay_output}")

# Create visualization 3: Just the warped video on court (no blend)
warped_only = court_img.copy()
# Create mask for video content
mask = np.any(video_warped > 0, axis=2).astype(np.uint8) * 255
warped_only = np.where(mask[:,:,None] > 0, video_warped, court_img)

warped_output = calib_dir / "calibration_verification_warped.jpg"
cv2.imwrite(str(warped_output), warped_only)
print(f"✓ Saved: {warped_output}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE!")
print("="*70)
print(f"\nImages saved to: {calib_dir}")
print("1. calibration_verification_points.jpg - Side-by-side with numbered points")
print("2. calibration_verification_overlay.jpg - Video warped and blended with court")
print("3. calibration_verification_warped.jpg - Video warped onto court")
print("\nOpen these images to verify your 1080p calibration!")
print("="*70)
