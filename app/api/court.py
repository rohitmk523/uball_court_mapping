"""API endpoints for court geometry."""
from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import COURT_IMAGE_FILE
from app.core.models import CourtBounds, CourtGeometry
from app.services import dxf_parser

router = APIRouter()


@router.get("/geometry", response_model=dict)
async def get_court_geometry():
    """Get court geometry as GeoJSON-like format."""
    geometry = dxf_parser.get_court_geometry()
    return dxf_parser.court_to_geojson(geometry)


@router.get("/bounds", response_model=CourtBounds)
async def get_court_bounds():
    """Get court bounding box."""
    geometry = dxf_parser.get_court_geometry()
    return geometry.bounds


@router.get("/image")
async def get_court_image():
    """Serve rendered court image."""
    # Generate image if it doesn't exist
    if not COURT_IMAGE_FILE.exists():
        dxf_parser.generate_court_image()

    return FileResponse(
        COURT_IMAGE_FILE,
        media_type="image/png",
        filename="court.png"
    )


@router.post("/regenerate-image")
async def regenerate_court_image(margin: float = 200.0, scale: float = 2.0):
    """
    Regenerate court image with custom parameters.

    Args:
        margin: Margin around court in cm (default: 200)
        scale: Pixels per cm (default: 2)
    """
    img_path = dxf_parser.generate_court_image(margin=margin, scale=scale)

    return {
        "status": "success",
        "image_path": str(img_path),
        "message": f"Court image regenerated with margin={margin}cm, scale={scale}px/cm"
    }
