"""
API endpoints for video processing with UWB tag overlays.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import logging
import uuid
from datetime import datetime
import io
import cv2

from ..services.video_processor import VideoProcessor
from ..services.calibration_integration import CalibrationIntegration
from ..services.tag_overlay import TagOverlay
from ..services.video_generator import VideoGenerator
from ..services.court_video_generator import CourtVideoGenerator
from ..core.court_config import CourtVideoConfig
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Track processing jobs
processing_jobs = {}


class VideoProcessRequest(BaseModel):
    """Request model for video processing."""
    video_file: str
    duration_seconds: Optional[float] = None
    start_time: float = 0.0
    sync_offset: float = 0.0
    show_trails: bool = True
    show_info: bool = True


class CourtVideoRequest(BaseModel):
    """Request model for court-centric video processing."""
    video_file: str
    sync_offset: Optional[float] = None
    auto_sync: bool = True
    tagging_threshold_cm: float = 200.0
    yolo_confidence: float = 0.5
    yolo_device: str = "mps"
    start_frame: int = 0
    max_frames: Optional[int] = None
    output_fps: float = 29.97


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    output_file: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


def process_video_task(
    job_id: str,
    video_file: str,
    duration: Optional[float],
    start_time: float,
    sync_offset: float,
    show_trails: bool,
    show_info: bool
):
    """Background task for processing video."""
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['message'] = 'Initializing...'

        # Initialize components
        video_processor = VideoProcessor(video_file)
        calibration = CalibrationIntegration(settings.CALIBRATION_FILE)
        tag_overlay = TagOverlay()

        generator = VideoGenerator(
            video_processor,
            calibration,
            tag_overlay,
            sync_offset
        )

        # Load tag data
        processing_jobs[job_id]['message'] = 'Loading tag data...'
        generator.load_tag_data(settings.TAGS_DIR)

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = settings.OUTPUT_DIR / f"video_with_tags_{timestamp}.mp4"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Process video
        processing_jobs[job_id]['message'] = 'Processing video...'
        success = generator.process_video(
            str(output_file),
            duration,
            start_time,
            show_trails,
            show_info
        )

        if success:
            processing_jobs[job_id]['status'] = 'completed'
            processing_jobs[job_id]['output_file'] = str(output_file)
            processing_jobs[job_id]['message'] = 'Video processing completed'
            processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        else:
            raise Exception("Video processing failed")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['message'] = str(e)
        processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()

    finally:
        if 'video_processor' in locals():
            video_processor.close()


@router.post("/process", response_model=JobStatus)
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Process video with UWB tag overlays.

    Args:
        request: Video processing request
        background_tasks: FastAPI background tasks

    Returns:
        Job status with job_id for tracking
    """
    # Validate video file
    video_path = Path(request.video_file)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_file}")

    # Create job
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        'job_id': job_id,
        'status': 'pending',
        'progress': 0.0,
        'message': 'Job queued',
        'output_file': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None
    }

    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id,
        request.video_file,
        request.duration_seconds,
        request.start_time,
        request.sync_offset,
        request.show_trails,
        request.show_info
    )

    return JobStatus(**processing_jobs[job_id])


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get status of a processing job.

    Args:
        job_id: Job identifier

    Returns:
        Job status
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatus(**processing_jobs[job_id])


@router.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download processed video.

    Args:
        job_id: Job identifier

    Returns:
        Video file
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = processing_jobs[job_id]

    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")

    output_file = job.get('output_file')
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_file,
        media_type="video/mp4",
        filename=Path(output_file).name
    )


@router.get("/preview/{frame_number}")
async def get_preview_frame(
    frame_number: int,
    video_file: str,
    sync_offset: float = 0.0
):
    """
    Get preview of a single frame with tag overlays.

    Args:
        frame_number: Frame number to preview
        video_file: Path to video file
        sync_offset: Sync offset in seconds

    Returns:
        JPEG image of frame with tags
    """
    try:
        # Initialize components
        video_processor = VideoProcessor(video_file)
        calibration = CalibrationIntegration(settings.CALIBRATION_FILE)
        tag_overlay = TagOverlay()

        generator = VideoGenerator(
            video_processor,
            calibration,
            tag_overlay,
            sync_offset
        )

        # Load tag data
        generator.load_tag_data(settings.TAGS_DIR)

        # Generate preview
        frame = generator.generate_preview_frame(frame_number, show_trails=False)

        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(io_buf, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'video_processor' in locals():
            video_processor.close()


@router.get("/list-jobs")
async def list_jobs():
    """List all processing jobs."""
    return {
        "jobs": [
            JobStatus(**job) for job in processing_jobs.values()
        ]
    }


@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output file."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = processing_jobs[job_id]

    # Delete output file if exists
    output_file = job.get('output_file')
    if output_file:
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to delete output file: {e}")

    # Remove job
    del processing_jobs[job_id]

    return {"message": f"Job {job_id} deleted"}


# ============================================================================
# Court-Centric Video Processing Endpoints
# ============================================================================


def process_court_view_task(job_id: str, config: CourtVideoConfig):
    """Background task for court-centric video processing."""
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['message'] = 'Initializing court video generator...'

        # Create generator
        generator = CourtVideoGenerator(config)

        # Process video
        processing_jobs[job_id]['message'] = 'Processing video frames...'
        result = generator.process_video()

        if result.success:
            processing_jobs[job_id]['status'] = 'completed'
            processing_jobs[job_id]['output_file'] = result.output_video_path
            processing_jobs[job_id]['events_file'] = result.output_events_path
            processing_jobs[job_id]['events_count'] = result.events_count
            processing_jobs[job_id]['message'] = f'Processing completed. {result.events_count} events logged.'
            processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()
            processing_jobs[job_id]['progress'] = 1.0
        else:
            raise Exception(result.error_message)

    except Exception as e:
        logger.error(f"Court video job {job_id} failed: {e}", exc_info=True)
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['message'] = str(e)
        processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()


@router.post("/process-court-view", response_model=JobStatus)
async def process_court_view(request: CourtVideoRequest, background_tasks: BackgroundTasks):
    """
    Process video with court-centric visualization.

    This endpoint creates a video where the basketball court is the primary canvas,
    and players are projected from video onto the court with UWB tags.

    Args:
        request: Court video processing request
        background_tasks: FastAPI background tasks

    Returns:
        Job status with job_id for tracking
    """
    # Validate video file
    video_path = Path(request.video_file)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_file}")

    # Create configuration
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = CourtVideoConfig(
            video_path=request.video_file,
            sync_offset_seconds=request.sync_offset,
            auto_sync=request.auto_sync,
            yolo_confidence=request.yolo_confidence,
            yolo_device=request.yolo_device,
            tagging_threshold_cm=request.tagging_threshold_cm,
            start_frame=request.start_frame,
            max_frames=request.max_frames,
            output_fps=request.output_fps,
            output_video_path=f"data/output/court_view_{timestamp}.mp4",
            output_events_path=f"data/output/tag_events_{timestamp}.json"
        )

        # Validate configuration
        is_valid, error_msg = config.validate()
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {error_msg}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")

    # Create job
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        'job_id': job_id,
        'status': 'pending',
        'progress': 0.0,
        'message': 'Job queued',
        'output_file': None,
        'events_file': None,
        'events_count': 0,
        'created_at': datetime.now().isoformat(),
        'completed_at': None
    }

    # Start background processing
    background_tasks.add_task(process_court_view_task, job_id, config)

    return JobStatus(**processing_jobs[job_id])


@router.get("/download-events/{job_id}")
async def download_events(job_id: str):
    """
    Download tag events JSON for a completed court video job.

    Args:
        job_id: Job identifier

    Returns:
        Events JSON file
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = processing_jobs[job_id]

    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")

    events_file = job.get('events_file')
    if not events_file or not Path(events_file).exists():
        raise HTTPException(status_code=404, detail="Events file not found")

    return FileResponse(
        events_file,
        media_type="application/json",
        filename=Path(events_file).name
    )
