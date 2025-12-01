"""FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.api import tags, court, calibration, tracking
from app.core.config import ensure_directories

# Ensure directories exist
ensure_directories()

# Initialize FastAPI app
app = FastAPI(
    title="Basketball Court Tracking System",
    description="UWB tag tracking and YOLO-based player detection with court calibration",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include API routers
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(court.router, prefix="/api/court", tags=["Court"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["Calibration"])
app.include_router(tracking.router, prefix="/api/tracking", tags=["Tracking"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main visualization page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/calibration", response_class=HTMLResponse)
async def calibration_page(request: Request):
    """Calibration UI page."""
    return templates.TemplateResponse("calibration.html", {"request": request})


@app.get("/tracking", response_class=HTMLResponse)
async def tracking_page(request: Request):
    """Tracking visualization page."""
    return templates.TemplateResponse("tracking.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
