import cv2
import numpy as np
import json
from pathlib import Path

class CalibrationValidationTool:
    """
    Three-step validation tool:
    1. Click on court image
    2. Click on original video frame
    3. Click on warped video (to verify transformation)
    """
    def __init__(self):
        # Load current calibration
        with open("data/calibration/homography.json") as f:
            calib = json.load(f)

        self.H = np.array(calib["homography"])

        # Load images
        self.court_img = cv2.imread("data/calibration/court_image.png")
        self.video_img = cv2.imread("video_frame_100.jpg")

        # Create warped video
        # H maps Court -> Video. We need Video -> Court.
        # Use WARP_INVERSE_MAP because H is Dst -> Src.
        self.warped_img = cv2.warpPerspective(
            self.video_img,
            self.H,
            (self.court_img.shape[1], self.court_img.shape[0]),
            flags=cv2.WARP_INVERSE_MAP
        )

        # Save warped for reference
        cv2.imwrite("warped_video_frame.jpg", self.warped_img)
        print("Saved warped video as 'warped_video_frame.jpg'")

        # Validation points
        self.court_points = []
        self.video_points = []
        self.warped_points = []

        self.current_mode = 'court'  # 'court' -> 'video' -> 'warped'
        self.window_name = "Validation Tool - Press SPACE to skip, ESC to finish"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'court':
                self.court_points.append([x, y])
                print(f"✓ Court point {len(self.court_points)}: ({x}, {y})")
                self.current_mode = 'video'

            elif self.current_mode == 'video':
                self.video_points.append([x, y])
                print(f"✓ Video point {len(self.video_points)}: ({x}, {y})")
                self.current_mode = 'warped'

            elif self.current_mode == 'warped':
                self.warped_points.append([x, y])
                print(f"✓ Warped point {len(self.warped_points)}: ({x}, {y})")
                print(f"→ Point set {len(self.warped_points)} complete!\n")
                self.current_mode = 'court'  # Back to start

    def get_display_image(self):
        """Get current image based on mode"""
        if self.current_mode == 'court':
            img = self.court_img.copy()
            # Draw existing points
            for pt in self.court_points:
                cv2.circle(img, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(img, str(len(self.court_points) + 1),
                           (pt[0] + 15, pt[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Instructions
            text = f"COURT IMAGE - Click point #{len(self.court_points) + 1}"
            cv2.putText(img, text, (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            return img

        elif self.current_mode == 'video':
            img = self.video_img.copy()
            # Draw existing points
            for pt in self.video_points:
                cv2.circle(img, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(img, str(len(self.video_points) + 1),
                           (pt[0] + 15, pt[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Instructions
            text = f"VIDEO FRAME - Click corresponding point #{len(self.video_points) + 1}"
            cv2.putText(img, text, (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            return img

        elif self.current_mode == 'warped':
            img = self.warped_img.copy()
            # Draw existing points
            for pt in self.warped_points:
                cv2.circle(img, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(img, str(len(self.warped_points) + 1),
                           (pt[0] + 15, pt[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Instructions
            text = f"WARPED VIDEO - Click validation point #{len(self.warped_points) + 1}"
            cv2.putText(img, text, (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
            return img

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "="*70)
        print("CALIBRATION VALIDATION TOOL")
        print("="*70)
        print("\nWorkflow:")
        print("  1. Click on COURT image (green)")
        print("  2. Click corresponding point on VIDEO frame (yellow)")
        print("  3. Click validation point on WARPED video (magenta)")
        print("  4. Repeat for multiple points")
        print("\nControls:")
        print("  - SPACE: Skip current step")
        print("  - ESC: Finish and analyze")
        print("="*70 + "\n")

        while True:
            display = self.get_display_image()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                # Skip current step
                if self.current_mode == 'court':
                    self.current_mode = 'video'
                elif self.current_mode == 'video':
                    self.current_mode = 'warped'
                elif self.current_mode == 'warped':
                    self.current_mode = 'court'

        cv2.destroyAllWindows()
        self.analyze_results()

    def analyze_results(self):
        """Analyze validation results"""
        print("\n" + "="*70)
        print("VALIDATION ANALYSIS")
        print("="*70)

        if len(self.court_points) == 0:
            print("\nNo validation points collected.")
            return

        print(f"\nCollected {len(self.court_points)} validation point sets")
        print(f"\n{'Point':<8} {'Court (x,y)':<25} {'Video (x,y)':<25} {'Warped (x,y)':<25} {'Error (px)'}")
        print("-"*110)

        errors = []
        for i in range(len(self.court_points)):
            court_pt = np.array(self.court_points[i])
            video_pt = np.array(self.video_points[i]) if i < len(self.video_points) else None
            warped_pt = np.array(self.warped_points[i]) if i < len(self.warped_points) else None

            if warped_pt is not None:
                # Calculate error between court point and warped validation point
                error = np.linalg.norm(court_pt - warped_pt)
                errors.append(error)

                print(f"{i+1:<8} {str(tuple(court_pt)):<25} "
                      f"{str(tuple(video_pt)) if video_pt is not None else 'N/A':<25} "
                      f"{str(tuple(warped_pt)):<25} {error:>6.1f}")

        if errors:
            print("\n" + "="*110)
            print(f"Average Error: {np.mean(errors):.1f} pixels")
            print(f"Max Error: {np.max(errors):.1f} pixels")
            print(f"Min Error: {np.min(errors):.1f} pixels")
            print("="*110)

            if np.mean(errors) > 50:
                print("\n⚠️  High average error! Calibration needs improvement.")
                print("   - Try recalibrating with more precise points")
                print("   - Focus on clear landmarks (corners, line intersections)")
            elif np.mean(errors) > 20:
                print("\n⚠️  Moderate error. Calibration could be better.")
            else:
                print("\n✓ Good calibration! Low average error.")

        print("\n")

if __name__ == "__main__":
    tool = CalibrationValidationTool()
    tool.run()
