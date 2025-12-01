/**
 * Calibration UI - Manual correspondence point selection for homography
 */

class CalibrationUI {
    constructor() {
        this.courtCanvas = document.getElementById('courtCanvas');
        this.videoCanvas = document.getElementById('videoCanvas');

        if (!this.courtCanvas || !this.videoCanvas) {
            console.error('Canvas elements not found');
            return;
        }

        this.courtCtx = this.courtCanvas.getContext('2d');
        this.videoCtx = this.videoCanvas.getContext('2d');

        this.courtImage = null;
        this.videoImage = null;

        this.courtPoints = [];
        this.videoPoints = [];

        this.waitingForCourtPoint = false;
        this.waitingForVideoPoint = false;

        this.initializeUI();
        this.loadImages();
    }

    initializeUI() {
        // Load video frame button
        document.getElementById('loadVideoFrame')?.addEventListener('click', () => this.loadVideoFrame());

        // Clear points button
        document.getElementById('clearPoints')?.addEventListener('click', () => this.clearPoints());

        // Undo last point button
        document.getElementById('undoPoint')?.addEventListener('click', () => this.undoLastPoint());

        // Submit calibration button
        document.getElementById('submitCalibration')?.addEventListener('click', () => this.submitCalibration());

        // Canvas click handlers
        this.courtCanvas.addEventListener('click', (e) => this.handleCourtClick(e));
        this.videoCanvas.addEventListener('click', (e) => this.handleVideoClick(e));

        // Check calibration status
        this.checkCalibrationStatus();
    }

    async checkCalibrationStatus() {
        try {
            const response = await fetch('/api/calibration/status');
            const status = await response.json();

            if (status.calibrated) {
                document.getElementById('calibrationStatus').textContent =
                    `Calibrated (${status.num_points} points, ${new Date(status.timestamp).toLocaleString()})`;
                document.getElementById('calibrationStatus').style.color = 'green';
            } else {
                document.getElementById('calibrationStatus').textContent = 'Not calibrated';
                document.getElementById('calibrationStatus').style.color = 'red';
            }
        } catch (error) {
            console.error('Failed to check calibration status:', error);
        }
    }

    async loadImages() {
        try {
            // Load court image
            const courtResponse = await fetch('/api/calibration/court-image');
            const courtBlob = await courtResponse.blob();
            const courtUrl = URL.createObjectURL(courtBlob);

            this.courtImage = new Image();
            this.courtImage.onload = () => {
                this.drawCourtCanvas();
                URL.revokeObjectURL(courtUrl);
            };
            this.courtImage.src = courtUrl;

        } catch (error) {
            console.error('Failed to load court image:', error);
        }
    }

    async loadVideoFrame() {
        const frameNumber = document.getElementById('frameNumber').value;

        try {
            const response = await fetch(`/api/calibration/video-frame?frame=${frameNumber}`);
            const videoBlob = await response.blob();
            const videoUrl = URL.createObjectURL(videoBlob);

            this.videoImage = new Image();
            this.videoImage.onload = () => {
                this.drawVideoCanvas();
                URL.revokeObjectURL(videoUrl);
            };
            this.videoImage.src = videoUrl;

        } catch (error) {
            console.error('Failed to load video frame:', error);
            alert('Failed to load video frame. Please try a different frame number.');
        }
    }

    drawCourtCanvas() {
        if (!this.courtImage) return;

        // Set canvas size to match image
        this.courtCanvas.width = Math.min(this.courtImage.width, 800);
        this.courtCanvas.height = this.courtImage.height * (this.courtCanvas.width / this.courtImage.width);

        // Draw image
        this.courtCtx.clearRect(0, 0, this.courtCanvas.width, this.courtCanvas.height);
        this.courtCtx.drawImage(this.courtImage, 0, 0, this.courtCanvas.width, this.courtCanvas.height);

        // Draw points
        this.drawPoints(this.courtCtx, this.courtPoints, this.courtCanvas.width / this.courtImage.width);
    }

    drawVideoCanvas() {
        if (!this.videoImage) return;

        // Set canvas size to match image
        this.videoCanvas.width = Math.min(this.videoImage.width, 800);
        this.videoCanvas.height = this.videoImage.height * (this.videoCanvas.width / this.videoImage.width);

        // Draw image
        this.videoCtx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
        this.videoCtx.drawImage(this.videoImage, 0, 0, this.videoCanvas.width, this.videoCanvas.height);

        // Draw points
        this.drawPoints(this.videoCtx, this.videoPoints, this.videoCanvas.width / this.videoImage.width);
    }

    drawPoints(ctx, points, scale) {
        points.forEach((point, index) => {
            const x = point[0] * scale;
            const y = point[1] * scale;

            // Draw circle
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw number
            ctx.fillStyle = 'white';
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 3;
            ctx.font = 'bold 16px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.strokeText((index + 1).toString(), x, y - 15);
            ctx.fillText((index + 1).toString(), x, y - 15);
        });
    }

    handleCourtClick(event) {
        if (!this.courtImage) return;

        const rect = this.courtCanvas.getBoundingClientRect();
        const scaleX = this.courtImage.width / this.courtCanvas.width;
        const scaleY = this.courtImage.height / this.courtCanvas.height;

        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;

        // Check if we're waiting for a corresponding video point
        if (this.courtPoints.length > this.videoPoints.length) {
            alert('Please click on the video frame to mark the corresponding point first!');
            return;
        }

        this.courtPoints.push([x, y]);
        this.drawCourtCanvas();
        this.updateUI();

        // Prompt for video point
        if (this.courtPoints.length > this.videoPoints.length) {
            this.highlightCanvas(this.videoCanvas, true);
        }
    }

    handleVideoClick(event) {
        if (!this.videoImage) {
            alert('Please load a video frame first!');
            return;
        }

        const rect = this.videoCanvas.getBoundingClientRect();
        const scaleX = this.videoImage.width / this.videoCanvas.width;
        const scaleY = this.videoImage.height / this.videoCanvas.height;

        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;

        // Check if we're waiting for a corresponding court point
        if (this.videoPoints.length > this.courtPoints.length) {
            alert('Please click on the court image to mark the corresponding point first!');
            return;
        }

        this.videoPoints.push([x, y]);
        this.drawVideoCanvas();
        this.updateUI();

        // Remove highlight
        this.highlightCanvas(this.videoCanvas, false);

        // Prompt for next court point
        if (this.courtPoints.length === this.videoPoints.length) {
            this.highlightCanvas(this.courtCanvas, true);
            setTimeout(() => this.highlightCanvas(this.courtCanvas, false), 1000);
        }
    }

    highlightCanvas(canvas, highlight) {
        if (highlight) {
            canvas.style.border = '3px solid #3498db';
            canvas.style.boxShadow = '0 0 10px #3498db';
        } else {
            canvas.style.border = '1px solid #ddd';
            canvas.style.boxShadow = 'none';
        }
    }

    updateUI() {
        const numPoints = Math.min(this.courtPoints.length, this.videoPoints.length);
        document.getElementById('pointCount').textContent = numPoints;

        // Enable submit button if we have at least 10 points
        const submitBtn = document.getElementById('submitCalibration');
        if (submitBtn) {
            submitBtn.disabled = numPoints < 10;
        }
    }

    clearPoints() {
        this.courtPoints = [];
        this.videoPoints = [];
        this.drawCourtCanvas();
        this.drawVideoCanvas();
        this.updateUI();
    }

    undoLastPoint() {
        if (this.courtPoints.length > this.videoPoints.length) {
            this.courtPoints.pop();
        } else if (this.videoPoints.length > 0) {
            this.videoPoints.pop();
        }

        this.drawCourtCanvas();
        this.drawVideoCanvas();
        this.updateUI();
    }

    async submitCalibration() {
        const numPoints = Math.min(this.courtPoints.length, this.videoPoints.length);

        if (numPoints < 10) {
            alert('Please select at least 10 correspondence points.');
            return;
        }

        if (this.courtPoints.length !== this.videoPoints.length) {
            alert('Number of court and video points must match!');
            return;
        }

        try {
            const response = await fetch('/api/calibration/points', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    court_points: this.courtPoints,
                    video_points: this.videoPoints
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Calibration failed');
            }

            const result = await response.json();

            alert(`Calibration successful! ${numPoints} points used.\nHomography matrix computed.`);

            // Update status
            await this.checkCalibrationStatus();

        } catch (error) {
            console.error('Calibration error:', error);
            alert('Calibration failed: ' + error.message);
        }
    }
}

// Global instance
let calibrationUI = null;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        calibrationUI = new CalibrationUI();
    });
} else {
    calibrationUI = new CalibrationUI();
}

console.log('Calibration UI loaded');
