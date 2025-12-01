"""API endpoints for player tracking and tag matching."""
import json
import cv2
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.core.config import SYNC_FILE, VIDEO_FILE
from app.core.models import SyncPoint
from app.services import detector, tracker, matcher, calibration, log_parser

router = APIRouter()


@router.post("/sync")
async def set_sync_point(sync: SyncPoint):
    """
    Set manual synchronization point between video and UWB timestamps.

    Args:
        sync: SyncPoint with video_frame and uwb_timestamp

    Returns:
        Confirmation message
    """
    try:
        # Save sync point
        SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(SYNC_FILE, 'w') as f:
            json.dump(sync.model_dump(), f, indent=2)

        return {
            "status": "success",
            "message": f"Sync point set: frame {sync.video_frame} = timestamp {sync.uwb_timestamp}",
            "sync": sync.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save sync point: {str(e)}")


@router.get("/sync", response_model=Optional[SyncPoint])
async def get_sync_point():
    """Get current sync point configuration."""
    if not SYNC_FILE.exists():
        return None

    try:
        with open(SYNC_FILE, 'r') as f:
            data = json.load(f)
            return SyncPoint(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sync point: {str(e)}")


@router.delete("/sync")
async def delete_sync_point():
    """Delete saved sync point."""
    if SYNC_FILE.exists():
        SYNC_FILE.unlink()
        return {"status": "success", "message": "Sync point deleted"}
    else:
        raise HTTPException(status_code=404, detail="Sync point not found")


@router.post("/process")
async def process_video(
    start_frame: Optional[int] = 0,
    end_frame: Optional[int] = None,
    frame_skip: int = 5,
    conf_threshold: float = 0.5
):
    """
    Process video with YOLO detection and tracking.

    Args:
        start_frame: Starting frame (default: 0)
        end_frame: Ending frame (None = process all)
        frame_skip: Process every Nth frame (default: 5)
        conf_threshold: YOLO confidence threshold (default: 0.5)

    Returns:
        Processing status and results summary
    """
    try:
        # Check if calibration exists
        if not calibration.is_calibrated():
            raise HTTPException(
                status_code=400,
                detail="Calibration required. Please calibrate first at /calibration"
            )

        # Check if sync point exists
        if not SYNC_FILE.exists():
            raise HTTPException(
                status_code=400,
                detail="Sync point required. Please set sync point first."
            )

        # Get video info
        cap = cv2.VideoCapture(str(VIDEO_FILE))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if end_frame is None:
            end_frame = total_frames

        # Get detector
        det = detector.get_detector(conf_threshold=conf_threshold)

        # Process video with YOLO
        detections_per_frame = det.process_video_batch(
            start_frame=start_frame,
            end_frame=end_frame,
            frame_skip=frame_skip,
            cache=True
        )

        # Apply tracking
        track = tracker.get_tracker()
        tracks_per_frame = track.track_video(detections_per_frame)

        return {
            "status": "success",
            "message": f"Processed {len(detections_per_frame)} frames",
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "processed_frames": list(detections_per_frame.keys())
            },
            "detection_summary": {
                "frames_with_detections": len([f for f, d in detections_per_frame.items() if len(d) > 0]),
                "total_detections": sum(len(d) for d in detections_per_frame.values()),
                "avg_detections_per_frame": sum(len(d) for d in detections_per_frame.values()) / max(len(detections_per_frame), 1)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/results/{frame}")
async def get_frame_results(frame: int):
    """
    Get detection and tracking results for a specific frame.

    Args:
        frame: Frame number

    Returns:
        Detection and tracking results for the frame
    """
    try:
        # Get detector and tracker
        det = detector.get_detector()
        track = tracker.get_tracker()

        # Try to get from cache
        # For now, return a message to process video first
        return {
            "status": "info",
            "message": "Please process video first using POST /api/tracking/process",
            "frame": frame
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/matched/{frame}")
async def get_matched_tags(frame: int):
    """
    Get tag-to-player matches for a specific frame.

    Args:
        frame: Frame number

    Returns:
        Tag-to-player matches for the frame
    """
    try:
        # Check if calibration exists
        if not calibration.is_calibrated():
            raise HTTPException(status_code=400, detail="Calibration required")

        # Check if sync point exists
        if not SYNC_FILE.exists():
            raise HTTPException(status_code=400, detail="Sync point required")

        # Load sync point
        with open(SYNC_FILE, 'r') as f:
            sync_data = json.load(f)
            sync = SyncPoint(**sync_data)

        # Load homography
        H = calibration.get_homography_matrix()
        if H is None:
            raise HTTPException(status_code=400, detail="Calibration not found")

        # Get video FPS
        cap = cv2.VideoCapture(str(VIDEO_FILE))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Calculate timestamp for this frame
        frame_offset = frame - sync.video_frame
        time_per_frame = 1000000 / fps  # microseconds
        timestamp = sync.uwb_timestamp + int(frame_offset * time_per_frame)

        # Get tags at this timestamp
        tags = log_parser.get_tags_at_timestamp(timestamp, tolerance=500000)

        # For now, return tag positions
        # Full matching would require processed tracks
        return {
            "status": "success",
            "frame": frame,
            "timestamp": timestamp,
            "tags": [{"tag_id": getattr(tag, 'tag_id', 0), "x": tag.x, "y": tag.y} for tag in tags],
            "message": "Process video first for full tracking and matching"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
