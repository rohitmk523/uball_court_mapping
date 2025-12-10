#!/usr/bin/env python
"""
Run calibration for 1080p video.
Saves results to data/calibration/1080p/
"""

import sys
from pathlib import Path

# Import the CalibrationTool
from calibration_tool import CalibrationTool

def main():
    """Run 1080p calibration."""
    # Paths for 1080p
    video_path = sys.argv[1] if len(sys.argv) > 1 else "GX020018_1080p.MP4"
    court_image_path = "data/calibration/court_image.png"
    output_dir = "data/calibration/1080p"

    print("="*60)
    print("BASKETBALL COURT CALIBRATION TOOL - 1080p")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Court: {court_image_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    print("\nInstructions:")
    print("1. Click on the COURT image to mark a point")
    print("2. Then click the corresponding point on the VIDEO")
    print("3. Repeat for at least 4 points (15+ recommended)")
    print("4. Click 'Save Calibration' when done")
    print("="*60)
    print("\nStarting GUI...")

    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tool = CalibrationTool(video_path, court_image_path, output_dir)
        tool.run()

        print("\n" + "="*60)
        print("Calibration saved to:", output_dir)
        print("Files created:")
        print("  - homography.json")
        print("  - court_image.png")
        print("  - video_frame.png")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
