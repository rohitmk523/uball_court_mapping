#!/usr/bin/env python3
"""Estimate and correct camera lens distortion using court lines"""

import cv2
import numpy as np
from pathlib import Path

# Load the video frame
video_frame = cv2.imread("video_frame_100.jpg")
h, w = video_frame.shape[:2]

print(f"Video frame size: {w}x{h}")

# Estimate camera matrix (assume camera center at image center)
# Focal length estimated from image dimensions
fx = fy = w  # Initial guess
cx = w / 2
cy = h / 2

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

print("\nInitial camera matrix:")
print(camera_matrix)

# Try different distortion coefficients
# [k1, k2, p1, p2, k3] where:
# k1, k2, k3 = radial distortion
# p1, p2 = tangential distortion

# For GoPro/action cameras, typical k1 is around -0.2 to -0.4
distortion_configs = [
    ("No distortion", np.zeros(5)),
    ("Light barrel", np.array([-0.2, 0.0, 0, 0, 0])),
    ("Medium barrel", np.array([-0.3, 0.05, 0, 0, 0])),
    ("Heavy barrel", np.array([-0.4, 0.1, 0, 0, 0])),
    ("GoPro-like", np.array([-0.35, 0.15, 0, 0, 0])),
]

# Create undistorted versions
for name, dist_coeffs in distortion_configs:
    print(f"\nGenerating: {name}")
    print(f"  Distortion coeffs: {dist_coeffs}")

    # Undistort
    undistorted = cv2.undistort(video_frame, camera_matrix, dist_coeffs)

    # Save
    filename = f"video_frame_100_undistorted_{name.replace(' ', '_').lower()}.jpg"
    cv2.imwrite(filename, undistorted)
    print(f"  Saved: {filename}")

print("\n" + "="*60)
print("Created multiple undistorted versions!")
print("Please check each one and tell me which looks best.")
print("The court lines should appear straight, not curved.")
print("="*60)

# Also create a comparison image showing original vs undistorted
fig_width = w // 2
fig_height = h // 2

# Original
original_small = cv2.resize(video_frame, (fig_width, fig_height))

# Medium barrel (most likely)
medium_undist = cv2.undistort(video_frame, camera_matrix,
                               np.array([-0.3, 0.05, 0, 0, 0]))
medium_small = cv2.resize(medium_undist, (fig_width, fig_height))

# Stack horizontally
comparison = np.hstack([original_small, medium_small])

# Add labels
cv2.putText(comparison, "ORIGINAL (Distorted)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(comparison, "UNDISTORTED (Medium)", (fig_width + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imwrite("distortion_comparison.jpg", comparison)
print("\nSaved comparison: distortion_comparison.jpg")
