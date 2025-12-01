import json
import cv2
import numpy as np

# Load calibration
with open("data/calibration/homography.json") as f:
    calib = json.load(f)

court_pts = np.array(calib["court_points"], dtype=np.float32)
video_pts = np.array(calib["video_points"], dtype=np.float32)
H = np.array(calib["homography"])

# Recompute to get mask
H_check, mask = cv2.findHomography(video_pts, court_pts, cv2.RANSAC, 5.0)

print(f"\n{'Point':<6} {'Court (x,y)':<20} {'Video (x,y)':<20} {'Status'}")
print("="*70)
for i in range(len(court_pts)):
    status = "INLIER ✓" if mask[i][0] else "OUTLIER ✗"
    print(f"{i+1:<6} {str(tuple(court_pts[i].astype(int))):<20} {str(tuple(video_pts[i].astype(int))):<20} {status}")

print(f"\n{'='*70}")
print(f"Total: {len(court_pts)} points | Inliers: {np.sum(mask)} | Outliers: {len(mask) - np.sum(mask)}")
print(f"{'='*70}\n")

print("Recommendation:")
print("  - Points marked OUTLIER ✗ are bad - they don't fit the transformation")
print("  - Try recalibrating with only 10-12 VERY PRECISE points")
print("  - Focus on clear, unambiguous landmarks (corners, line intersections)")
print("  - Spread points across the ENTIRE visible court area")
