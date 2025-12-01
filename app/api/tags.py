"""API endpoints for tag data."""
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.core.models import ProcessingStatus, TagData, TagPosition
from app.services import log_parser

router = APIRouter()


@router.post("/process", response_model=ProcessingStatus)
async def process_log_file():
    """
    Trigger log file processing to generate tag JSON files.

    This endpoint parses the UWB log file and creates individual JSON files
    for each tag in the data/tags/ directory.
    """
    try:
        result = log_parser.process_log()

        return ProcessingStatus(
            status="complete",
            progress=1.0,
            message=f"Successfully processed {result['tags_processed']} tags",
            result=result
        )
    except Exception as e:
        return ProcessingStatus(
            status="error",
            progress=0.0,
            message=f"Error processing log file: {str(e)}",
            result=None
        )


@router.get("/status")
async def get_processing_status():
    """Check if tags have been processed."""
    tag_ids = log_parser.list_tag_ids()

    if tag_ids:
        return {
            "processed": True,
            "num_tags": len(tag_ids),
            "tag_ids": tag_ids
        }
    else:
        return {
            "processed": False,
            "num_tags": 0,
            "tag_ids": []
        }


@router.get("/list", response_model=List[int])
async def list_tags():
    """Get list of all available tag IDs."""
    return log_parser.list_tag_ids()


@router.get("/{tag_id}", response_model=TagData)
async def get_tag(tag_id: int):
    """Get all positions for a specific tag."""
    tag_data = log_parser.get_tag_data(tag_id)

    if not tag_data:
        raise HTTPException(status_code=404, detail=f"Tag {tag_id} not found")

    return tag_data


@router.get("/at/{timestamp}", response_model=List[TagPosition])
async def get_tags_at_timestamp(timestamp: int, tolerance: int = 100):
    """
    Get all tag positions at a specific timestamp.

    Args:
        timestamp: UWB timestamp
        tolerance: Timestamp tolerance in UWB units (default: 100)
    """
    positions = log_parser.get_tags_at_timestamp(timestamp, tolerance=tolerance)
    return positions


@router.get("/timerange/", response_model=dict)
async def get_tags_in_timerange(start: int, end: int):
    """
    Get all tag positions within a time range.

    Args:
        start: Start UWB timestamp
        end: End UWB timestamp

    Returns:
        Dictionary mapping tag_id to list of positions
    """
    result = log_parser.get_tags_in_timerange(start, end)

    # Convert to JSON-serializable format
    return {
        str(tag_id): [pos.model_dump() for pos in positions]
        for tag_id, positions in result.items()
    }
