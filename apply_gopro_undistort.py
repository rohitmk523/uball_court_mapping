import cv2
import numpy as np

# Load the ORIGINAL distorted frame
video_frame = cv2.imread("video_frame_100_original_distorted.jpg")
h, w = video_frame.shape[:2]

# Camera matrix
fx = fy = w
cx = w / 2
cy = h / 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# DSL606 lens distortion coefficients (approx -19% barrel distortion)
dist_coeffs = np.array([-0.19, 0.0, 0, 0, 0])

# Undistort
undistorted = cv2.undistort(video_frame, camera_matrix, dist_coeffs)

# Save as the main video_frame_100.jpg
cv2.imwrite("video_frame_100.jpg", undistorted)
print("✓ Applied GoPro undistortion to video_frame_100.jpg")
print(f"  Original size: {video_frame.shape}")
print(f"  Undistorted size: {undistorted.shape}")
