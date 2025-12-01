"""API endpoints for calibration."""
import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import CALIBRATION_DIR, COURT_IMAGE_FILE, HOMOGRAPHY_FILE, VIDEO_FILE, VIDEO_FRAME_FILE
from app.core.models import Calibration, CalibrationPoints
from app.services import calibration as calib_service
from app.services import dxf_parser

router = APIRouter()


@router.get("/court-image")
async def get_court_image():
    """Serve court image for calibration."""
    # Generate if doesn't exist
    if not COURT_IMAGE_FILE.exists():
        dxf_parser.generate_court_image()

    return FileResponse(
        COURT_IMAGE_FILE,
        media_type="image/png",
        filename="court.png"
    )


@router.get("/video-frame")
async def get_video_frame(frame: int = 100):
    """
    Extract and serve a video frame for calibration.

    Args:
        frame: Frame number to extract (default: 100)
    """
    # Extract frame from video
    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video file")

    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    # Read frame
    ret, frame_img = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=400, detail=f"Failed to read frame {frame}")

    # Save frame
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(VIDEO_FRAME_FILE), frame_img)

    return FileResponse(
        VIDEO_FRAME_FILE,
        media_type="image/png",
        filename="video_frame.png"
    )


@router.post("/points", response_model=Calibration)
async def save_calibration_points(points: CalibrationPoints):
    """
    Save calibration points and compute homography.

    Args:
        points: Calibration correspondence points

    Returns:
        Calibration object with homography matrix
    """
    try:
        # Compute homography
        H = calib_service.compute_homography(
            points.court_points,
            points.video_points
        )

        # Save calibration
        calibration = calib_service.save_calibration(
            H,
            points.court_points,
            points.video_points
        )

        return calibration

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")


@router.get("/status")
async def get_calibration_status():
    """Check if calibration exists."""
    is_calibrated = calib_service.is_calibrated()

    if is_calibrated:
        calibration = calib_service.load_calibration()
        return {
            "calibrated": True,
            "timestamp": calibration.timestamp if calibration else None,
            "num_points": len(calibration.court_points) if calibration else 0
        }
    else:
        return {
            "calibrated": False,
            "timestamp": None,
            "num_points": 0
        }


@router.get("/homography", response_model=Calibration)
async def get_homography():
    """Get saved homography matrix."""
    calibration = calib_service.load_calibration()

    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")

    return calibration


@router.delete("/calibration")
async def delete_calibration():
    """Delete saved calibration."""
    if HOMOGRAPHY_FILE.exists():
        HOMOGRAPHY_FILE.unlink()
        return {"status": "success", "message": "Calibration deleted"}
    else:
        raise HTTPException(status_code=404, detail="Calibration not found")
