import cv2
import numpy as np

# Create test image
img = np.ones((1000, 2000, 3), dtype=np.uint8) * 255

# Center line
center_x = 1000

# Right arc - test reference
right_center = (1400, 500)
radius = 300

# Draw right arc from 90 to 270 (left side of circle toward center)
cv2.ellipse(img, right_center, (radius, radius), 0, 90, 270, (255, 0, 0), 3)

# Mirror left arc
left_center = (2 * center_x - right_center[0], right_center[1])  # (600, 500)

# Mirror angles: 90 → 90 (180-90=90), 270 → -90 (180-270=-90) = 270
# Actually: 90 → 270-90 = 180-(-90) = 270, 270 → 180-(-90) = 90
# Simpler: angles 90-270 mirrored become 270-90
cv2.ellipse(img, left_center, (radius, radius), 0, 270, 90, (0, 255, 0), 3)

# Draw center line
cv2.line(img, (center_x, 0), (center_x, 1000), (0, 0, 0), 2)

# Draw centers
cv2.circle(img, right_center, 5, (255, 0, 0), -1)
cv2.circle(img, left_center, 5, (0, 255, 0), -1)

cv2.imwrite('/Users/rohitkale/Cellstrat/GitHub_Repositories/uball_court_mapping/test_mirror.png', img)
print("Test image created: test_mirror.png")
print(f"Right arc: center={right_center}, angles 90-270 (blue)")
print(f"Left arc: center={left_center}, angles 270-90 (green)")
