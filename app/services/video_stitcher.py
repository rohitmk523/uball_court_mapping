"""
Video Stitcher for Side-by-Side Visualization.

Creates combined frames: [Video | Court Canvas]
- Left panel: Original video with detections (colors by UWB tag_id)
- Right panel: Court canvas with UWB dots + player projections

Reuses rendering logic from validation_overlap.py.

Usage:
    stitcher = VideoStitcher(court_image_path, court_dxf_path)

    # In frame loop
    combined_frame = stitcher.create_stitched_frame(
        video_frame, tracked_players, uwb_positions, id_mapper
    )
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from app.services.dxf_parser import parse_court_dxf
from app.services.persistent_id_mapper import PersistentIDMapper


# Colors (BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 100, 0)
COLOR_LIGHT_BLUE = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)


class VideoStitcher:
    """Stitches video frames with court canvas for dual-panel visualization."""

    def __init__(
        self,
        court_image_path: str = "data/calibration/court_image.png",
        court_dxf_path: str = "court_2.dxf",
        output_height: int = 1080,
        proximity_radius_cm: float = 200.0
    ):
        """
        Initialize video stitcher with court canvas.

        Args:
            court_image_path: Path to court image
            court_dxf_path: Path to court DXF file (for geometry)
            output_height: Output video height (default: 1080)
            proximity_radius_cm: UWB proximity radius in cm (default: 200)
        """
        # Load court image and rotate to vertical (90° CW)
        self.court_img_orig = cv2.imread(court_image_path)
        if self.court_img_orig is None:
            raise FileNotFoundError(f"Court image not found: {court_image_path}")

        self.court_canvas_vertical = cv2.rotate(
            self.court_img_orig,
            cv2.ROTATE_90_CLOCKWISE
        )
        self.vert_h, self.vert_w = self.court_canvas_vertical.shape[:2]

        # Setup geometry transform
        self.output_height = output_height
        self._setup_geometry(court_dxf_path, proximity_radius_cm)

        print(f"✅ VideoStitcher initialized")
        print(f"   Court canvas: {self.vert_w}x{self.vert_h} (vertical)")
        print(f"   Output height: {self.output_height}")
        print(f"   Proximity radius: {proximity_radius_cm}cm ({self.radius_200cm_px}px)")

    def _setup_geometry(self, court_dxf_path: str, proximity_radius_cm: float):
        """
        Setup UWB world → screen coordinate transform.

        Args:
            court_dxf_path: Path to court DXF file
            proximity_radius_cm: Proximity radius in cm
        """
        # Parse court geometry
        geometry = parse_court_dxf(court_dxf_path)
        self.court_bounds = geometry.bounds

        # Court dimensions (cm)
        court_width = self.court_bounds.max_x - self.court_bounds.min_x  # Long side
        court_height = self.court_bounds.max_y - self.court_bounds.min_y  # Short side

        # Calculate scale (fit court to canvas with padding)
        PADDING = 50
        scale_y = (self.vert_h - PADDING * 2) / court_width
        scale_x = (self.vert_w - PADDING * 2) / court_height

        self.scale = min(scale_x, scale_y)

        # Calculate offsets to center court
        scaled_long = court_width * self.scale
        scaled_short = court_height * self.scale

        self.offset_y = (self.vert_h - scaled_long) / 2
        self.offset_x = (self.vert_w - scaled_short) / 2

        # Convert proximity radius to pixels
        self.radius_200cm_px = int(proximity_radius_cm * self.scale)

        # Calculate dot size (scale with canvas width)
        self.DOT_RADIUS = int(40 * (self.vert_w / 7341 * 1.7))
        if self.DOT_RADIUS < 15:
            self.DOT_RADIUS = 15
        self.DOT_OUTLINE = 3

    def uwb_to_vertical_screen(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        """
        Convert UWB world coordinates (cm) to vertical canvas pixels.

        Coordinate mapping:
        - Court X (long side) → Canvas Y (height)
        - Court Y (short side) → Canvas X (width)

        Args:
            x_cm: UWB X coordinate (cm)
            y_cm: UWB Y coordinate (cm)

        Returns:
            (sx, sy) screen coordinates in pixels
        """
        sy = ((x_cm - self.court_bounds.min_x) * self.scale) + self.offset_y
        sx = ((y_cm - self.court_bounds.min_y) * self.scale) + self.offset_x
        return int(sx), int(sy)

    def create_stitched_frame(
        self,
        video_frame: np.ndarray,
        tracked_players: List[Dict],
        uwb_positions: Dict[int, Tuple[float, float]],
        id_mapper: Optional['PersistentIDMapper'] = None
    ) -> np.ndarray:
        """
        Create side-by-side stitched frame: [Video | Court Canvas].

        Args:
            video_frame: Original video frame (already has detections drawn)
            tracked_players: List of tracked players with court coords
            uwb_positions: Dict of {tag_id: (x_cm, y_cm)} UWB positions
            id_mapper: PersistentIDMapper for color lookup (optional, for single-pass mode)

        Returns:
            Combined frame [1920px video | scaled canvas]
        """
        # Prepare canvas
        canvas = self.court_canvas_vertical.copy()

        # Convert UWB positions to screen coordinates
        uwb_screen = {}
        for tag_id, (ux, uy) in uwb_positions.items():
            sx, sy = self.uwb_to_vertical_screen(ux, uy)
            uwb_screen[tag_id] = (sx, sy)

        # Draw UWB proximity circles (semi-transparent)
        overlay = canvas.copy()
        for tag_id, (sx, sy) in uwb_screen.items():
            cv2.circle(
                overlay,
                (sx, sy),
                self.radius_200cm_px,
                COLOR_LIGHT_BLUE,
                2
            )
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw UWB dots (blue with white outline)
        for tag_id, (sx, sy) in uwb_screen.items():
            cv2.circle(canvas, (sx, sy), self.DOT_RADIUS, COLOR_BLUE, -1)
            cv2.circle(canvas, (sx, sy), self.DOT_RADIUS, COLOR_WHITE, self.DOT_OUTLINE)

        # Draw projected players (color by tag_id)
        for player in tracked_players:
            if 'court_x' not in player or 'court_y' not in player:
                continue
            if player['court_x'] is None or player['court_y'] is None:
                continue

            cx, cy = int(player['court_x']), int(player['court_y'])
            tag_id = player.get('uwb_tag_id')

            # Determine color (by tag_id for persistence)
            if id_mapper:
                color = id_mapper.get_color_for_tag(tag_id)
            else:
                # Fallback: random color by tag_id (single-pass mode)
                if tag_id is not None:
                    np.random.seed(tag_id)
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                else:
                    color = COLOR_RED

            # Check if within proximity of assigned tag
            if tag_id and tag_id in uwb_screen:
                tx, ty = uwb_screen[tag_id]
                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                if dist <= self.radius_200cm_px:
                    color = COLOR_GREEN  # Inside proximity
                else:
                    color = COLOR_RED  # Tag assigned but drifted
            elif tag_id is None:
                color = COLOR_RED  # No tag assigned
            # else: use color from id_mapper

            # Draw player dot
            cv2.circle(canvas, (cx, cy), self.DOT_RADIUS, color, -1)
            cv2.circle(canvas, (cx, cy), self.DOT_RADIUS, COLOR_WHITE, self.DOT_OUTLINE)

        # Resize canvas to match output height
        court_scale = self.output_height / self.vert_h
        court_panel_w = int(self.vert_w * court_scale)
        canvas_resized = cv2.resize(canvas, (court_panel_w, self.output_height))

        # Stitch horizontally: [video | canvas]
        total_width = 1920 + court_panel_w
        combined = np.zeros((self.output_height, total_width, 3), dtype=np.uint8)

        # Place video frame (left panel)
        combined[0:1080, 0:1920] = video_frame

        # Place court canvas (right panel)
        combined[0:1080, 1920:total_width] = canvas_resized

        return combined
