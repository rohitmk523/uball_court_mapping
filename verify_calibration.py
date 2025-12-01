#!/usr/bin/env python3
"""Verify calibration by visualizing the points and transformation"""

import cv2
import numpy as np
import json
from pathlib import Path

# Load calibration data
calib_file = Path("data/calibration/homography.json")
with open(calib_file) as f:
    calib_data = json.load(f)

court_points = np.array(calib_data["court_points"], dtype=np.float32)
video_points = np.array(calib_data["video_points"], dtype=np.float32)
H = np.array(calib_data["homography"], dtype=np.float32)

print(f"Loaded calibration with {len(court_points)} point pairs")
print(f"Inliers: {calib_data.get('inliers', 'N/A')}")
print(f"Outliers: {calib_data.get('outliers', 'N/A')}")

# Load images
court_img = cv2.imread("data/calibration/court_image.png")
# Rotate court image 90 degrees clockwise to match calibration tool
court_img = cv2.rotate(court_img, cv2.ROTATE_90_CLOCKWISE)
print("Rotated court image 90 degrees clockwise")
video_img = cv2.imread("video_frame_100.jpg")

# Create visualization 1: Side-by-side with points marked
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
cv2.putText(side_by_side, "COURT", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.putText(side_by_side, "COURT", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

cv2.putText(side_by_side, "VIDEO", (court_width + 50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.putText(side_by_side, "VIDEO", (court_width + 50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

cv2.imwrite("calibration_verification_points.jpg", side_by_side)
print("\nSaved: calibration_verification_points.jpg")

# Create visualization 2: Warped video overlaid on court
# H maps Court -> Video. warpPerspective expects Src -> Dst (Video -> Court) by default.
# So we must use WARP_INVERSE_MAP to tell it H is Dst -> Src.
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

cv2.imwrite("calibration_verification_overlay.jpg", overlay)
print("Saved: calibration_verification_overlay.jpg")

# Create visualization 3: Just the warped video on court (no blend)
warped_only = court_img.copy()
# Create mask for video content
mask = np.any(video_warped > 0, axis=2).astype(np.uint8) * 255
warped_only = np.where(mask[:,:,None] > 0, video_warped, court_img)

cv2.imwrite("calibration_verification_warped.jpg", warped_only)
print("Saved: calibration_verification_warped.jpg")

print("\nVerification images created:")
print("1. calibration_verification_points.jpg - Side-by-side with numbered points")
print("2. calibration_verification_overlay.jpg - Video warped and blended with court")
print("3. calibration_verification_warped.jpg - Video warped onto court")
print("\nOpen these images to verify your calibration!")
