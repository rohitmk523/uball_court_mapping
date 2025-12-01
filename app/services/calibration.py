"""Calibration service for computing homography matrix."""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from app.core.config import HOMOGRAPHY_FILE
from app.core.models import Calibration, CalibrationPoints


def compute_homography(
    court_points: List[List[float]],
    video_points: List[List[float]]
) -> np.ndarray:
    """
    Compute homography matrix from correspondence points.

    Args:
        court_points: List of [x, y] points in court coordinates (cm)
        video_points: List of [x, y] points in video coordinates (pixels)

    Returns:
        3x3 homography matrix as numpy array

    Raises:
        ValueError: If insufficient points or computation fails
    """
    if len(court_points) < 4 or len(video_points) < 4:
        raise ValueError("At least 4 correspondence points are required")

    if len(court_points) != len(video_points):
        raise ValueError("Number of court and video points must match")

    # Convert to numpy arrays
    court_pts = np.array(court_points, dtype=np.float32)
    video_pts = np.array(video_points, dtype=np.float32)

    # Compute homography using RANSAC for robustness
    H, mask = cv2.findHomography(court_pts, video_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Failed to compute homography matrix")

    return H


def transform_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Transform a single point using homography matrix.

    Args:
        point: (x, y) coordinates
        H: 3x3 homography matrix

    Returns:
        Transformed (x, y) coordinates
    """
    # Convert to homogeneous coordinates
    pt = np.array([point[0], point[1], 1.0])

    # Apply homography
    transformed = H @ pt

    # Convert back from homogeneous coordinates
    if transformed[2] != 0:
        x = transformed[0] / transformed[2]
        y = transformed[1] / transformed[2]
    else:
        x, y = transformed[0], transformed[1]

    return (float(x), float(y))


def transform_points(points: List[Tuple[float, float]], H: np.ndarray) -> List[Tuple[float, float]]:
    """
    Transform multiple points using homography matrix.

    Args:
        points: List of (x, y) coordinates
        H: 3x3 homography matrix

    Returns:
        List of transformed (x, y) coordinates
    """
    return [transform_point(pt, H) for pt in points]


def inverse_transform_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Transform a point using inverse homography (video to court).

    Args:
        point: (x, y) coordinates in video space
        H: 3x3 homography matrix (court to video)

    Returns:
        Transformed (x, y) coordinates in court space
    """
    H_inv = np.linalg.inv(H)
    return transform_point(point, H_inv)


def save_calibration(
    H: np.ndarray,
    court_points: List[List[float]],
    video_points: List[List[float]],
    output_path: Path = HOMOGRAPHY_FILE
) -> Calibration:
    """
    Save calibration data to JSON file.

    Args:
        H: 3x3 homography matrix
        court_points: List of court correspondence points
        video_points: List of video correspondence points
        output_path: Path to save calibration file

    Returns:
        Calibration object
    """
    calibration = Calibration(
        homography=H.tolist(),
        court_points=court_points,
        video_points=video_points,
        timestamp=datetime.now().isoformat()
    )

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(calibration.model_dump(), f, indent=2)

    return calibration


def load_calibration(calibration_path: Path = HOMOGRAPHY_FILE) -> Calibration | None:
    """
    Load calibration data from JSON file.

    Args:
        calibration_path: Path to calibration file

    Returns:
        Calibration object or None if file doesn't exist
    """
    if not calibration_path.exists():
        return None

    with open(calibration_path, 'r') as f:
        data = json.load(f)
        return Calibration(**data)


def get_homography_matrix(calibration_path: Path = HOMOGRAPHY_FILE) -> np.ndarray | None:
    """
    Get homography matrix from saved calibration.

    Args:
        calibration_path: Path to calibration file

    Returns:
        3x3 numpy array or None if calibration doesn't exist
    """
    calibration = load_calibration(calibration_path)

    if calibration is None:
        return None

    return np.array(calibration.homography, dtype=np.float64)


def is_calibrated(calibration_path: Path = HOMOGRAPHY_FILE) -> bool:
    """
    Check if calibration exists.

    Args:
        calibration_path: Path to calibration file

    Returns:
        True if calibration file exists, False otherwise
    """
    return calibration_path.exists()
