"""Parse UWB log file and generate tag JSON files."""
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.core.config import LOG_FILE, TAGS_DIR
from app.core.models import TagData, TagPosition


def parse_log_line(line: str) -> tuple[str, int, float, float, int] | None:
    """
    Parse a single log line.

    Format: 2025-11-22 02:26:40.578 | Tag 1587672 | X=454 | Y=469 | Timestamp=268737020

    Returns:
        Tuple of (datetime_str, tag_id, x, y, uwb_timestamp) or None if parsing fails
    """
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| Tag (\d+) \| X=(\d+) \| Y=(\d+) \| Timestamp=(\d+)'
    match = re.match(pattern, line.strip())

    if match:
        datetime_str = match.group(1)
        tag_id = int(match.group(2))
        x = float(match.group(3))
        y = float(match.group(4))
        uwb_timestamp = int(match.group(5))
        return (datetime_str, tag_id, x, y, uwb_timestamp)

    return None


def parse_log_file(log_path: Path = LOG_FILE) -> Dict[int, List[TagPosition]]:
    """
    Parse the entire log file and group by tag_id.

    Args:
        log_path: Path to the log file

    Returns:
        Dictionary mapping tag_id to list of TagPosition objects
    """
    tags_data = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                datetime_str, tag_id, x, y, uwb_timestamp = parsed
                position = TagPosition(
                    timestamp=uwb_timestamp,
                    x=x,
                    y=y,
                    datetime=datetime_str
                )
                tags_data[tag_id].append(position)

    # Sort positions by timestamp for each tag
    for tag_id in tags_data:
        tags_data[tag_id].sort(key=lambda p: p.timestamp)

    return dict(tags_data)


def generate_tag_files(tags_data: Dict[int, List[TagPosition]], output_dir: Path = TAGS_DIR) -> int:
    """
    Generate individual JSON files for each tag.

    Args:
        tags_data: Dictionary mapping tag_id to list of positions
        output_dir: Directory to write tag JSON files

    Returns:
        Number of tags processed
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag_id, positions in tags_data.items():
        tag_data = TagData(tag_id=tag_id, positions=positions)
        output_file = output_dir / f"{tag_id}.json"

        with open(output_file, 'w') as f:
            json.dump(tag_data.model_dump(), f, indent=2)

    return len(tags_data)


def process_log(log_path: Path = LOG_FILE, output_dir: Path = TAGS_DIR) -> dict:
    """
    Complete processing pipeline: parse log and generate tag files.

    Args:
        log_path: Path to the log file
        output_dir: Directory to write tag JSON files

    Returns:
        Dictionary with processing statistics
    """
    tags_data = parse_log_file(log_path)
    num_tags = generate_tag_files(tags_data, output_dir)

    # Calculate time range
    all_timestamps = []
    for positions in tags_data.values():
        all_timestamps.extend([p.timestamp for p in positions])

    if all_timestamps:
        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)
    else:
        min_ts = max_ts = 0

    return {
        "tags_processed": num_tags,
        "time_range": [min_ts, max_ts],
        "total_positions": sum(len(positions) for positions in tags_data.values())
    }


def get_tag_data(tag_id: int, tags_dir: Path = TAGS_DIR) -> TagData | None:
    """
    Load tag data from JSON file.

    Args:
        tag_id: Tag ID to load
        tags_dir: Directory containing tag JSON files

    Returns:
        TagData object or None if file doesn't exist
    """
    tag_file = tags_dir / f"{tag_id}.json"

    if not tag_file.exists():
        return None

    with open(tag_file, 'r') as f:
        data = json.load(f)
        return TagData(**data)


def list_tag_ids(tags_dir: Path = TAGS_DIR) -> List[int]:
    """
    List all available tag IDs.

    Args:
        tags_dir: Directory containing tag JSON files

    Returns:
        List of tag IDs
    """
    if not tags_dir.exists():
        return []

    tag_ids = []
    for file_path in tags_dir.glob("*.json"):
        try:
            tag_id = int(file_path.stem)
            tag_ids.append(tag_id)
        except ValueError:
            continue

    return sorted(tag_ids)


def get_tags_at_timestamp(timestamp: int, tags_dir: Path = TAGS_DIR, tolerance: int = 100) -> List[TagPosition]:
    """
    Get all tag positions at a specific timestamp (with tolerance).

    Args:
        timestamp: UWB timestamp to query
        tags_dir: Directory containing tag JSON files
        tolerance: Timestamp tolerance in UWB units

    Returns:
        List of TagPosition objects at the given timestamp
    """
    tag_ids = list_tag_ids(tags_dir)
    positions = []

    for tag_id in tag_ids:
        tag_data = get_tag_data(tag_id, tags_dir)
        if not tag_data:
            continue

        # Find closest position within tolerance
        for position in tag_data.positions:
            if abs(position.timestamp - timestamp) <= tolerance:
                positions.append(position)
                break

    return positions


def get_tags_in_timerange(start_ts: int, end_ts: int, tags_dir: Path = TAGS_DIR) -> Dict[int, List[TagPosition]]:
    """
    Get all tag positions within a time range.

    Args:
        start_ts: Start UWB timestamp
        end_ts: End UWB timestamp
        tags_dir: Directory containing tag JSON files

    Returns:
        Dictionary mapping tag_id to list of positions in range
    """
    tag_ids = list_tag_ids(tags_dir)
    result = {}

    for tag_id in tag_ids:
        tag_data = get_tag_data(tag_id, tags_dir)
        if not tag_data:
            continue

        # Filter positions in range
        positions_in_range = [
            p for p in tag_data.positions
            if start_ts <= p.timestamp <= end_ts
        ]

        if positions_in_range:
            result[tag_id] = positions_in_range

    return result
