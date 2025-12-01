"""Tag-to-player matching service."""
import numpy as np
from typing import Dict, List, Optional, Tuple

from app.core.config import TAG_MATCH_THRESHOLD_CM
from app.core.models import TagMatch, TagPosition, Track
from app.services import calibration as calib_service


def transform_tags_to_video(
    tags: List[TagPosition],
    H: np.ndarray
) -> List[Tuple[int, Tuple[float, float]]]:
    """
    Transform tag positions from court to video coordinates.

    Args:
        tags: List of TagPosition objects in court coordinates (cm)
        H: Homography matrix (court to video)

    Returns:
        List of tuples (tag_id, (video_x, video_y))
    """
    transformed = []

    for tag in tags:
        # Transform point from court to video
        video_pos = calib_service.transform_point((tag.x, tag.y), H)
        transformed.append((tag.tag_id if hasattr(tag, 'tag_id') else 0, video_pos))

    return transformed


def calculate_distance_cm(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Distance in same units as input
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_closest_tag(
    player_pos_video: Tuple[float, float],
    tags: List[TagPosition],
    H: np.ndarray,
    threshold_cm: float = TAG_MATCH_THRESHOLD_CM
) -> Optional[Tuple[int, float]]:
    """
    Find closest tag to a player position (in court coordinates).

    Args:
        player_pos_video: Player position in video coordinates (pixels)
        tags: List of TagPosition objects in court coordinates (cm)
        H: Homography matrix (court to video)
        threshold_cm: Distance threshold in cm

    Returns:
        Tuple of (tag_id, distance_cm) or None if no match within threshold
    """
    # Transform player position from video to court coordinates
    player_pos_court = calib_service.inverse_transform_point(player_pos_video, H)

    closest_tag_id = None
    closest_distance = float('inf')

    for tag in tags:
        # Calculate distance in court coordinates (cm)
        distance = calculate_distance_cm(player_pos_court, (tag.x, tag.y))

        if distance < closest_distance:
            closest_distance = distance
            closest_tag_id = tag.tag_id if hasattr(tag, 'tag_id') else None

    # Check if within threshold
    if closest_distance <= threshold_cm:
        return (closest_tag_id, closest_distance)
    else:
        return None


def match_frame(
    tracks: List[Track],
    tags: List[TagPosition],
    H: np.ndarray,
    threshold_cm: float = TAG_MATCH_THRESHOLD_CM
) -> List[TagMatch]:
    """
    Match tags to tracked players for a single frame.

    Args:
        tracks: List of Track objects (players)
        tags: List of TagPosition objects
        H: Homography matrix (court to video)
        threshold_cm: Distance threshold for matching

    Returns:
        List of TagMatch objects
    """
    matches = []

    for track in tracks:
        # Get player position (bottom center of bbox)
        player_pos = track.bbox_bottom_center

        # Find closest tag
        result = find_closest_tag(player_pos, tags, H, threshold_cm)

        if result:
            tag_id, distance = result
            match = TagMatch(
                track_id=track.track_id,
                tag_id=tag_id,
                distance_cm=distance
            )
        else:
            # No match found
            match = TagMatch(
                track_id=track.track_id,
                tag_id=None,
                distance_cm=None
            )

        matches.append(match)

    return matches


def process_video_matching(
    tracks_per_frame: Dict[int, List[Track]],
    tags_per_timestamp: Dict[int, List[TagPosition]],
    H: np.ndarray,
    frame_to_timestamp: Dict[int, int],
    threshold_cm: float = TAG_MATCH_THRESHOLD_CM
) -> Dict[int, Dict]:
    """
    Process video matching for all frames.

    Args:
        tracks_per_frame: Dictionary mapping frame_id to tracks
        tags_per_timestamp: Dictionary mapping uwb_timestamp to tags
        H: Homography matrix
        frame_to_timestamp: Dictionary mapping frame_id to uwb_timestamp
        threshold_cm: Distance threshold

    Returns:
        Dictionary with frame_id -> {"tracks": [...], "tags": [...], "matches": [...]}
    """
    results = {}

    for frame_id, tracks in tracks_per_frame.items():
        # Get corresponding timestamp for this frame
        if frame_id not in frame_to_timestamp:
            continue

        timestamp = frame_to_timestamp[frame_id]

        # Get tags at this timestamp
        if timestamp not in tags_per_timestamp:
            # No tags at this timestamp
            matches = [
                TagMatch(track_id=track.track_id, tag_id=None, distance_cm=None)
                for track in tracks
            ]
            results[frame_id] = {
                "tracks": tracks,
                "tags": [],
                "matches": matches
            }
            continue

        tags = tags_per_timestamp[timestamp]

        # Perform matching
        matches = match_frame(tracks, tags, H, threshold_cm)

        results[frame_id] = {
            "tracks": tracks,
            "tags": tags,
            "matches": matches
        }

    return results


def create_frame_to_timestamp_mapping(
    sync_frame: int,
    sync_timestamp: int,
    video_fps: float,
    frame_ids: List[int]
) -> Dict[int, int]:
    """
    Create mapping from video frame IDs to UWB timestamps.

    Args:
        sync_frame: Video frame number at sync point
        sync_timestamp: UWB timestamp at sync point
        video_fps: Video frames per second
        frame_ids: List of frame IDs to map

    Returns:
        Dictionary mapping frame_id to uwb_timestamp
    """
    mapping = {}

    # Calculate time per frame in UWB timestamp units
    # Assuming UWB timestamps are in microseconds and roughly linear
    time_per_frame = 1000000 / video_fps  # microseconds per frame

    for frame_id in frame_ids:
        frame_offset = frame_id - sync_frame
        timestamp = sync_timestamp + int(frame_offset * time_per_frame)
        mapping[frame_id] = timestamp

    return mapping
