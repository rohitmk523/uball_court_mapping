#!/usr/bin/env python3
"""Sequential Calibration Tool - Click on one image, then the other"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class SequentialCalibrationTool:
    def __init__(self, video_path, court_image_path, output_dir):
        self.video_path = Path(video_path)
        self.court_image_path = Path(court_image_path)
        self.output_dir = Path(output_dir)

        # Load images
        print("Loading court image...")
        self.court_img = cv2.imread(str(court_image_path))
        if self.court_img is None:
            raise ValueError(f"Failed to load court image: {court_image_path}")

        print("Loading video frame 100...")
        # Load the pre-extracted (and undistorted) frame instead of from video
        self.video_img = cv2.imread("video_frame_100.jpg")
        if self.video_img is None:
            raise ValueError("Failed to load video_frame_100.jpg")

        # Point storage
        self.court_points = []
        self.video_points = []
        self.current_mode = 'court'  # 'court' or 'video'

        # Display image copies
        self.court_display = self.court_img.copy()
        self.video_display = self.video_img.copy()

        # Window name
        self.window_name = 'Calibration Tool - Sequential'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'court':
                # Add court point
                self.court_points.append([x, y])
                # Draw marker
                cv2.circle(self.court_display, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(self.court_display, str(len(self.court_points)),
                           (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Court point {len(self.court_points)}: ({x}, {y})")

                # Switch to video mode if we have more court points than video points
                if len(self.court_points) > len(self.video_points):
                    self.current_mode = 'video'
                    print(">>> Now click the corresponding point on the VIDEO")

            else:  # video mode
                # Add video point
                self.video_points.append([x, y])
                # Draw marker
                cv2.circle(self.video_display, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(self.video_display, str(len(self.video_points)),
                           (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Video point {len(self.video_points)}: ({x}, {y})")

                # Switch back to court mode
                self.current_mode = 'court'
                print(f">>> Point pair {len(self.video_points)} saved!")
                print(">>> Click next point on the COURT (or press 's' to save)")

    def get_current_display(self):
        """Get the current image to display based on mode"""
        if self.current_mode == 'court':
            img = self.court_display.copy()
            # Add instruction text
            text = f"COURT IMAGE - Click point #{len(self.court_points)+1}"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 0, 255), 3)
            cv2.putText(img, f"Points: {len(self.court_points)} court, {len(self.video_points)} video",
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            img = self.video_display.copy()
            # Add instruction text
            text = f"VIDEO IMAGE - Click corresponding point #{len(self.video_points)+1}"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 255, 0), 3)
            cv2.putText(img, f"Points: {len(self.court_points)} court, {len(self.video_points)} video",
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

    def run(self):
        """Run the calibration tool"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("=" * 60)
        print("SEQUENTIAL CALIBRATION TOOL")
        print("=" * 60)
        print("Instructions:")
        print("1. Click a point on the COURT image")
        print("2. Then click the corresponding point on the VIDEO image")
        print("3. Repeat for 10-15 point pairs")
        print()
        print("Controls:")
        print("  s = SAVE calibration (min 10 points)")
        print("  u = UNDO last point pair")
        print("  d = DELETE all points")
        print("  q = QUIT without saving")
        print("=" * 60)
        print()
        print(">>> Click first point on the COURT")

        while True:
            # Display current image
            display_img = self.get_current_display()
            cv2.imshow(self.window_name, display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting without saving...")
                break

            elif key == ord('s'):
                if len(self.court_points) >= 10 and len(self.video_points) >= 10:
                    self.save_calibration()
                    break
                else:
                    print(f"Need at least 10 point pairs! Currently have {len(self.video_points)}")

            elif key == ord('u'):
                # Undo last point pair
                if self.video_points:
                    self.video_points.pop()
                    # Redraw video display
                    self.video_display = self.video_img.copy()
                    for i, pt in enumerate(self.video_points):
                        cv2.circle(self.video_display, tuple(pt), 10, (0, 255, 0), -1)
                        cv2.putText(self.video_display, str(i+1),
                                   (pt[0] + 15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Undid last video point")

                if self.court_points:
                    self.court_points.pop()
                    # Redraw court display
                    self.court_display = self.court_img.copy()
                    for i, pt in enumerate(self.court_points):
                        cv2.circle(self.court_display, tuple(pt), 10, (0, 0, 255), -1)
                        cv2.putText(self.court_display, str(i+1),
                                   (pt[0] + 15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("Undid last court point")

                # Reset to court mode
                self.current_mode = 'court'
                print(">>> Click next point on the COURT")

            elif key == ord('d'):
                # Delete all points
                self.court_points = []
                self.video_points = []
                self.court_display = self.court_img.copy()
                self.video_display = self.video_img.copy()
                self.current_mode = 'court'
                print("Deleted all points")
                print(">>> Click first point on the COURT")

        cv2.destroyAllWindows()

    def save_calibration(self):
        """Compute and save homography"""
        print("\nComputing homography...")

        court_pts = np.array(self.court_points, dtype=np.float32)
        video_pts = np.array(self.video_points, dtype=np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(video_pts, court_pts, cv2.RANSAC, 5.0)

        inliers = np.sum(mask)
        outliers = len(mask) - inliers

        print(f"Homography computed!")
        print(f"Inliers: {inliers}/{len(mask)}")
        print(f"Outliers: {outliers}/{len(mask)}")

        # Save to JSON
        output_file = self.output_dir / "homography.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        calibration_data = {
            "homography": H.tolist(),
            "court_points": self.court_points,
            "video_points": self.video_points,
            "timestamp": datetime.now().isoformat(),
            "inliers": int(inliers),
            "outliers": int(outliers)
        }

        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"\nCalibration saved to: {output_file}")
        print(f"Used {len(self.court_points)} point pairs")


if __name__ == "__main__":
    tool = SequentialCalibrationTool(
        video_path="GX020018.MP4",
        court_image_path="data/calibration/court_image.png",
        output_dir="data/calibration"
    )
    tool.run()
