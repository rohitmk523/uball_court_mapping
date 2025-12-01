"""Parse DXF court file and generate court geometry."""
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import ezdxf
import numpy as np

from app.core.config import CALIBRATION_DIR, COURT_IMAGE_FILE, DXF_FILE
from app.core.models import CourtBounds, CourtGeometry


def parse_court_dxf(dxf_path: Path = DXF_FILE) -> CourtGeometry:
    """
    Parse DXF file and extract court geometry.

    Args:
        dxf_path: Path to the DXF file

    Returns:
        CourtGeometry object with all court elements
    """
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    polylines = []
    lines = []
    circles = []
    arcs = []

    # Extract all geometric entities
    for entity in modelspace:
        if entity.dxftype() == 'POLYLINE':
            # Get vertices of polyline
            vertices = []
            for vertex in entity.vertices:
                vertices.append((float(vertex.dxf.location.x), float(vertex.dxf.location.y)))
            if vertices:
                polylines.append(vertices)

        elif entity.dxftype() == 'LWPOLYLINE':
            # Lightweight polyline
            vertices = []
            for point in entity.get_points('xy'):
                vertices.append((float(point[0]), float(point[1])))
            if vertices:
                polylines.append(vertices)

        elif entity.dxftype() == 'LINE':
            # Single line segment
            start = entity.dxf.start
            end = entity.dxf.end
            lines.append((
                (float(start.x), float(start.y)),
                (float(end.x), float(end.y))
            ))

        elif entity.dxftype() == 'CIRCLE':
            # Circle
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            circles.append((
                (float(center.x), float(center.y)),
                radius
            ))

        elif entity.dxftype() == 'ARC':
            # Arc (for 3-point lines)
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            start_angle = float(entity.dxf.start_angle)
            end_angle = float(entity.dxf.end_angle)
            arcs.append((
                (float(center.x), float(center.y)),
                radius,
                start_angle,
                end_angle
            ))

    # Calculate bounding box
    all_points = []
    for polyline in polylines:
        all_points.extend(polyline)
    for line in lines:
        all_points.extend(line)
    for circle in circles:
        center, radius = circle
        all_points.extend([
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius)
        ])
    for arc in arcs:
        center, radius, _, _ = arc
        all_points.extend([
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius)
        ])

    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = max_x = min_y = max_y = 0

    bounds = CourtBounds(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        width=max_x - min_x,
        height=max_y - min_y
    )

    return CourtGeometry(
        polylines=polylines,
        lines=lines,
        circles=circles,
        arcs=arcs,
        bounds=bounds
    )


def render_court_to_image(
    court_geometry: CourtGeometry,
    output_path: Path = COURT_IMAGE_FILE,
    margin: float = 200.0,
    scale: float = 2.0
) -> np.ndarray:
    """
    Render court geometry to an image.

    Args:
        court_geometry: CourtGeometry object
        output_path: Path to save the image
        margin: Margin around court in cm (for tags outside court)
        scale: Pixels per cm

    Returns:
        Rendered image as numpy array
    """
    bounds = court_geometry.bounds

    # Calculate image dimensions with margin
    img_width = int((bounds.width + 2 * margin) * scale)
    img_height = int((bounds.height + 2 * margin) * scale)

    # Create white background
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Transform function from court coordinates to image coordinates
    def court_to_img(x: float, y: float) -> Tuple[int, int]:
        """Convert court coordinates to image pixel coordinates."""
        img_x = int((x - bounds.min_x + margin) * scale)
        # Flip Y axis (court Y-up, image Y-down)
        img_y = int((bounds.max_y - y + margin) * scale)
        return (img_x, img_y)

    # Draw polylines
    for polyline in court_geometry.polylines:
        pts = np.array([court_to_img(x, y) for x, y in polyline], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    # Draw lines
    for line in court_geometry.lines:
        pt1 = court_to_img(*line[0])
        pt2 = court_to_img(*line[1])
        cv2.line(img, pt1, pt2, color=(0, 0, 0), thickness=2)

    # Draw circles
    for circle in court_geometry.circles:
        center_court, radius = circle
        center_img = court_to_img(*center_court)
        radius_img = int(radius * scale)
        cv2.circle(img, center_img, radius_img, color=(0, 0, 0), thickness=2)

    # Draw arcs (3-point lines)
    # Use the right arc as reference and mirror it to create the left arc
    court_center_x = (bounds.min_x + bounds.max_x) / 2

    # Find the right arc (higher X coordinate)
    right_arc = None
    for arc in court_geometry.arcs:
        center_court, radius, start_angle, end_angle = arc
        if center_court[0] > court_center_x:
            right_arc = arc
            break

    if right_arc:
        # Draw right arc with standard transformation
        center_court, radius, start_angle, end_angle = right_arc
        center_img = court_to_img(*center_court)
        radius_img = int(radius * scale)

        # Standard Y-flip transformation
        start_angle_img = (360 - end_angle) % 360
        end_angle_img = (360 - start_angle) % 360

        cv2.ellipse(img, center_img, (radius_img, radius_img), 0,
                   start_angle_img, end_angle_img, (0, 0, 0), 2)

        # Mirror the right arc to create left arc
        # Mirror center point across court center line
        left_center_x = 2 * court_center_x - center_court[0]
        left_center_court = (left_center_x, center_court[1])
        left_center_img = court_to_img(*left_center_court)

        # Swap start/end angles for perfect mirror (as shown in test)
        left_start_angle = end_angle_img
        left_end_angle = start_angle_img

        # Ensure we draw in the correct direction (increasing angle)
        if left_end_angle <= left_start_angle:
            left_end_angle += 360

        # Use ellipse2Poly to manually generate arc points to ensure correct direction
        pts = cv2.ellipse2Poly(left_center_img, (radius_img, radius_img), 0,
                             int(left_start_angle), int(left_end_angle), 1)
        cv2.polylines(img, [pts], False, (0, 0, 0), 2)

    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    return img


def court_to_geojson(court_geometry: CourtGeometry) -> dict:
    """
    Convert court geometry to GeoJSON-like format for JavaScript rendering.

    Args:
        court_geometry: CourtGeometry object

    Returns:
        Dictionary with court data in JSON-serializable format
    """
    return {
        "type": "CourtGeometry",
        "bounds": court_geometry.bounds.model_dump(),
        "polylines": court_geometry.polylines,
        "lines": court_geometry.lines,
        "circles": court_geometry.circles,
        "arcs": court_geometry.arcs
    }


def get_court_geometry(dxf_path: Path = DXF_FILE) -> CourtGeometry:
    """
    Get court geometry (cached version).

    Args:
        dxf_path: Path to the DXF file

    Returns:
        CourtGeometry object
    """
    # For now, just parse directly
    # In future, could cache to file
    return parse_court_dxf(dxf_path)


def generate_court_image(
    dxf_path: Path = DXF_FILE,
    output_path: Path = COURT_IMAGE_FILE,
    margin: float = 200.0,
    scale: float = 2.0
) -> Path:
    """
    Generate court image from DXF file.

    Args:
        dxf_path: Path to the DXF file
        output_path: Path to save the image
        margin: Margin around court in cm
        scale: Pixels per cm

    Returns:
        Path to the generated image
    """
    court_geometry = parse_court_dxf(dxf_path)
    render_court_to_image(court_geometry, output_path, margin, scale)
    return output_path
