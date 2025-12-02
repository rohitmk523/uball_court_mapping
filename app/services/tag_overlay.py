"""
Tag overlay system for drawing UWB tags on video frames.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TagOverlay:
    """Handles drawing UWB tags on video frames."""

    def __init__(
        self,
        tag_size: int = 20,
        text_size: float = 0.8,
        text_thickness: int = 2,
        trail_length: int = 30
    ):
        """
        Initialize tag overlay.

        Args:
            tag_size: Radius of tag circle in pixels
            text_size: Font scale for tag ID text
            text_thickness: Thickness of tag ID text
            trail_length: Number of previous positions to show as trail
        """
        self.tag_size = tag_size
        self.text_size = text_size
        self.text_thickness = text_thickness
        self.trail_length = trail_length

        # Tag history for trails
        self.tag_history: Dict[str, List[Tuple[int, int]]] = {}

        # Color for UWB tags - BLUE (BGR format)
        self.uwb_tag_color = (255, 100, 0)  # Blue

    def get_tag_color(self, tag_id: str) -> Tuple[int, int, int]:
        """
        Get color for UWB tag (blue for floor position).

        Args:
            tag_id: Tag identifier

        Returns:
            BGR color tuple (blue for all UWB tags)
        """
        return self.uwb_tag_color

    def draw_tag(
        self,
        frame: np.ndarray,
        tag_id: str,
        pixel_x: int,
        pixel_y: int,
        show_trail: bool = True,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw a single tag on the frame.

        Args:
            frame: Video frame (BGR)
            tag_id: Tag identifier
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            show_trail: Whether to show movement trail
            color: Optional custom color (BGR), uses default if None

        Returns:
            Frame with tag drawn
        """
        if color is None:
            color = self.get_tag_color(tag_id)

        # Update tag history
        if tag_id not in self.tag_history:
            self.tag_history[tag_id] = []

        self.tag_history[tag_id].append((pixel_x, pixel_y))

        # Keep only recent history
        if len(self.tag_history[tag_id]) > self.trail_length:
            self.tag_history[tag_id] = self.tag_history[tag_id][-self.trail_length:]

        # Draw trail if enabled
        if show_trail and len(self.tag_history[tag_id]) > 1:
            trail_points = np.array(self.tag_history[tag_id], dtype=np.int32)
            for i in range(len(trail_points) - 1):
                alpha = i / len(trail_points)  # Fade older points
                thickness = max(1, int(3 * alpha))
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(
                    frame,
                    tuple(trail_points[i]),
                    tuple(trail_points[i + 1]),
                    trail_color,
                    thickness,
                    cv2.LINE_AA
                )

        # Draw tag circle
        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            self.tag_size,
            color,
            -1,  # Filled
            cv2.LINE_AA
        )

        # Draw white outline
        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            self.tag_size,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Draw tag ID text
        text = str(tag_id)
        text_offset_y = -self.tag_size - 8

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_size,
            self.text_thickness
        )

        # Draw text background (semi-transparent)
        text_x = pixel_x - text_width // 2
        text_y = pixel_y + text_offset_y

        # Background rectangle
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (text_x - 4, text_y - text_height - 4),
            (text_x + text_width + 4, text_y + baseline + 4),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_size,
            (255, 255, 255),
            self.text_thickness,
            cv2.LINE_AA
        )

        return frame

    def draw_tags(
        self,
        frame: np.ndarray,
        tags: List[Dict],
        show_trails: bool = True
    ) -> np.ndarray:
        """
        Draw multiple tags on the frame.

        Args:
            frame: Video frame (BGR)
            tags: List of tag dictionaries with keys:
                  - tag_id: Tag identifier
                  - pixel_x: X coordinate
                  - pixel_y: Y coordinate
                  - color: (optional) BGR color tuple
            show_trails: Whether to show movement trails

        Returns:
            Frame with tags drawn
        """
        for tag in tags:
            if 'tag_id' not in tag or 'pixel_x' not in tag or 'pixel_y' not in tag:
                logger.warning(f"Invalid tag data: {tag}")
                continue

            # Get custom color if provided
            color = tag.get('color', None)

            frame = self.draw_tag(
                frame,
                str(tag['tag_id']),
                tag['pixel_x'],
                tag['pixel_y'],
                show_trails,
                color
            )

        return frame

    def draw_info_panel(
        self,
        frame: np.ndarray,
        frame_number: int,
        time_seconds: float,
        num_tags: int,
        num_players: int = 0
    ) -> np.ndarray:
        """
        Draw information panel on frame.

        Args:
            frame: Video frame
            frame_number: Current frame number
            time_seconds: Current time in seconds
            num_tags: Number of visible UWB tags
            num_players: Number of detected players (YOLO)

        Returns:
            Frame with info panel
        """
        # Info panel background
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (400, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw info text
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        color = (255, 255, 255)

        info_lines = [
            f"Frame: {frame_number}",
            f"Time: {time_seconds:.2f}s",
            f"Players (YOLO): {num_players}",
            f"Tags (UWB): {num_tags}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (20, y_offset + i * 25),
                font,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA
            )

        return frame

    def reset_history(self):
        """Clear all tag history."""
        self.tag_history.clear()
        logger.info("Tag history cleared")

    def clear_tag_history(self, tag_id: str):
        """
        Clear history for a specific tag.

        Args:
            tag_id: Tag identifier
        """
        if tag_id in self.tag_history:
            del self.tag_history[tag_id]
