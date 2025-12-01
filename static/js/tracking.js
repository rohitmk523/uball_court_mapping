/**
 * Tracking Visualization - Display YOLO detections and tag matches
 */

class TrackingVisualizer {
    constructor() {
        this.videoCanvas = document.getElementById('videoCanvas');
        this.courtCanvas = document.getElementById('courtCanvas');

        if (!this.videoCanvas || !this.courtCanvas) {
            console.error('Canvas elements not found');
            return;
        }

        this.videoCtx = this.videoCanvas.getContext('2d');
        this.courtCtx = this.courtCanvas.getContext('2d');

        this.syncPoint = null;
        this.videoFPS = 30;
        this.processedFrames = [];
        this.currentFrameIndex = 0;
        this.isPlaying = false;
        this.animationFrameId = null;

        // Court renderer for synchronized view
        this.courtRenderer = null;

        this.initializeUI();
        this.loadSyncStatus();
        this.initializeCourtRenderer();
    }

    initializeUI() {
        // Sync button
        document.getElementById('setSyncBtn')?.addEventListener('click', () => this.setSyncPoint());

        // Process video button
        document.getElementById('processVideoBtn')?.addEventListener('click', () => this.processVideo());

        // Playback controls
        document.getElementById('playBtn')?.addEventListener('click', () => this.play());
        document.getElementById('pauseBtn')?.addEventListener('click', () => this.pause());
        document.getElementById('resetBtn')?.addEventListener('click', () => this.reset());

        // Timeline
        document.getElementById('videoTimeline')?.addEventListener('input', (e) => {
            const progress = parseFloat(e.target.value) / 100;
            this.currentFrameIndex = Math.floor(progress * (this.processedFrames.length - 1));
            this.updateFrame();
        });
    }

    async loadSyncStatus() {
        try {
            const response = await fetch('/api/tracking/sync');
            if (response.ok) {
                this.syncPoint = await response.json();
                if (this.syncPoint) {
                    document.getElementById('syncStatus').textContent =
                        `Synced: Frame ${this.syncPoint.video_frame} = Timestamp ${this.syncPoint.uwb_timestamp}`;
                    document.getElementById('syncStatus').style.color = 'green';
                }
            }
        } catch (error) {
            console.error('Failed to load sync status:', error);
        }
    }

    async setSyncPoint() {
        const frame = parseInt(document.getElementById('syncFrame').value);
        const timestamp = parseInt(document.getElementById('syncTimestamp').value);
        const status = document.getElementById('syncStatus');

        // Validate inputs
        const frameValidation = UIUtils.validateNumberInput(frame, 0);
        if (!frameValidation.valid) {
            status.textContent = 'Invalid frame: ' + frameValidation.message;
            status.style.color = 'red';
            return;
        }

        const timestampValidation = UIUtils.validateNumberInput(timestamp, 0);
        if (!timestampValidation.valid) {
            status.textContent = 'Invalid timestamp: ' + timestampValidation.message;
            status.style.color = 'red';
            return;
        }

        UIUtils.showInlineSpinner('syncStatus');

        try {
            const response = await UIUtils.fetchWithTimeout('/api/tracking/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_frame: frame,
                    uwb_timestamp: timestamp
                })
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.syncPoint = { video_frame: frame, uwb_timestamp: timestamp };
                status.textContent = `✓ Synced: Frame ${UIUtils.formatNumber(frame)} = Timestamp ${UIUtils.formatNumber(timestamp)}`;
                status.style.color = 'green';
            } else {
                status.textContent = 'Sync failed';
                status.style.color = 'red';
            }
        } catch (error) {
            UIUtils.handleAPIError(error);
            status.textContent = 'Error: ' + error.message;
            status.style.color = 'red';
        } finally {
            UIUtils.hideInlineSpinner('syncStatus');
        }
    }

    async processVideo() {
        const startFrame = parseInt(document.getElementById('startFrame').value);
        const endFrame = parseInt(document.getElementById('endFrame').value);
        const frameSkip = parseInt(document.getElementById('frameSkip').value);
        const status = document.getElementById('processingStatus');
        const btn = document.getElementById('processVideoBtn');

        // Validate inputs
        const startValidation = UIUtils.validateNumberInput(startFrame, 0);
        if (!startValidation.valid) {
            UIUtils.showStatus('processingStatus', 'Invalid start frame: ' + startValidation.message, 'error');
            return;
        }

        const endValidation = UIUtils.validateNumberInput(endFrame, startFrame);
        if (!endValidation.valid) {
            UIUtils.showStatus('processingStatus', 'Invalid end frame: ' + endValidation.message, 'error');
            return;
        }

        const skipValidation = UIUtils.validateNumberInput(frameSkip, 1, 30);
        if (!skipValidation.valid) {
            UIUtils.showStatus('processingStatus', 'Invalid frame skip: ' + skipValidation.message, 'error');
            return;
        }

        btn.disabled = true;
        const frameCount = Math.ceil((endFrame - startFrame) / frameSkip);
        UIUtils.showLoadingOverlay(
            'Processing video with YOLO...',
            `Processing ~${frameCount} frames. This may take several minutes.`
        );
        UIUtils.showStatus('processingStatus', 'Processing video with YOLO... This may take a few minutes.', 'info');

        try {
            const response = await UIUtils.fetchWithTimeout(
                `/api/tracking/process?start_frame=${startFrame}&end_frame=${endFrame}&frame_skip=${frameSkip}`,
                { method: 'POST' },
                300000  // 5 minute timeout for video processing
            );

            const result = await response.json();

            if (result.status === 'success') {
                UIUtils.showStatus('processingStatus', `✓ ${result.message}`, 'success');

                // Store processed frame IDs
                this.processedFrames = result.video_info.processed_frames;
                this.videoFPS = result.video_info.fps;

                document.getElementById('totalFrames').textContent = UIUtils.formatNumber(this.processedFrames.length);

                // Show detection summary
                const summary = result.detection_summary;
                if (summary) {
                    const summaryText = `Processed ${UIUtils.formatNumber(this.processedFrames.length)} frames. ` +
                        `Found ${UIUtils.formatNumber(summary.total_detections)} detections ` +
                        `(avg ${summary.avg_detections_per_frame.toFixed(1)} per frame)`;
                    UIUtils.showStatus('processingStatus', '✓ ' + summaryText, 'success');
                }

                // Enable playback controls
                UIUtils.enableElement('playBtn');
                UIUtils.enableElement('pauseBtn');
                UIUtils.enableElement('resetBtn');
                UIUtils.enableElement('videoTimeline');

                // Load first frame
                this.currentFrameIndex = 0;
                await this.updateFrame();

            } else {
                throw new Error(result.message || 'Processing failed');
            }

        } catch (error) {
            UIUtils.handleAPIError(error, 'processingStatus');
        } finally {
            UIUtils.hideLoadingOverlay();
            btn.disabled = false;
        }
    }

    async updateFrame() {
        if (this.processedFrames.length === 0) return;

        const frameId = this.processedFrames[this.currentFrameIndex];
        document.getElementById('currentFrame').textContent = frameId;

        try {
            // Get matched tags for this frame
            const response = await fetch(`/api/tracking/matched/${frameId}`);
            const data = await response.json();

            if (data.status === 'success') {
                document.getElementById('currentTimestamp').textContent = data.timestamp;
                document.getElementById('playerCount').textContent = data.tags.length; // For now, shows tag count
                document.getElementById('matchedTags').textContent = data.tags.length;

                // Update court visualization with tags
                if (this.courtRenderer) {
                    const tags = data.tags.map(t => ({
                        tag_id: t.tag_id,
                        x: t.x,
                        y: t.y
                    }));
                    this.courtRenderer.updateTags(tags);
                }

                // TODO: Load and display video frame with detections
                // This would require serving video frames from the API
            }

        } catch (error) {
            console.error('Failed to update frame:', error);
        }
    }

    play() {
        if (this.isPlaying || this.processedFrames.length === 0) return;
        this.isPlaying = true;
        this.animate();
    }

    pause() {
        this.isPlaying = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    reset() {
        this.pause();
        this.currentFrameIndex = 0;
        this.updateFrame();
        document.getElementById('videoTimeline').value = 0;
    }

    animate() {
        if (!this.isPlaying) return;

        // Advance frame
        this.currentFrameIndex++;
        if (this.currentFrameIndex >= this.processedFrames.length) {
            this.currentFrameIndex = 0; // Loop
        }

        this.updateFrame();

        // Update timeline
        const progress = (this.currentFrameIndex / (this.processedFrames.length - 1)) * 100;
        document.getElementById('videoTimeline').value = progress;

        // Continue animation at ~10 FPS for visualization
        setTimeout(() => {
            this.animationFrameId = requestAnimationFrame(() => this.animate());
        }, 100);
    }

    async initializeCourtRenderer() {
        // Wait for court renderer to be available
        const checkRenderer = setInterval(() => {
            if (window.CourtRenderer) {
                this.courtRenderer = new window.CourtRenderer('courtCanvas');
                clearInterval(checkRenderer);
            }
        }, 100);

        // Also inline a simple version if CourtRenderer not available
        setTimeout(() => {
            if (!this.courtRenderer) {
                this.courtRenderer = new SimpleCourtRenderer('courtCanvas');
            }
        }, 1000);
    }
}

// Simple court renderer for tracking page
class SimpleCourtRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.tags = [];
        this.courtBounds = { min_x: 0, min_y: 0, max_x: 2460, max_y: 1730 };

        this.canvas.width = 600;
        this.canvas.height = 400;

        this.loadCourtGeometry();
    }

    async loadCourtGeometry() {
        try {
            const response = await fetch('/api/court/geometry');
            const data = await response.json();
            this.courtGeometry = data;
            this.courtBounds = data.bounds;
            this.render();
        } catch (error) {
            console.error('Failed to load court geometry:', error);
        }
    }

    updateTags(tags) {
        this.tags = tags;
        this.render();
    }

    render() {
        if (!this.courtGeometry) return;

        // Clear
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#f8f9fa';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Simple court outline
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(20, 20, this.canvas.width - 40, this.canvas.height - 40);

        // Draw tags
        const scaleX = (this.canvas.width - 40) / this.courtBounds.width;
        const scaleY = (this.canvas.height - 40) / this.courtBounds.height;
        const scale = Math.min(scaleX, scaleY);

        this.tags.forEach(tag => {
            const x = 20 + tag.x * scale;
            const y = 20 + (this.courtBounds.height - tag.y) * scale;

            this.ctx.fillStyle = `hsl(${(tag.tag_id * 137.5) % 360}, 70%, 50%)`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
            this.ctx.fill();

            this.ctx.fillStyle = '#000';
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(tag.tag_id, x, y - 10);
        });
    }
}

// Global instance
let trackingVisualizer = null;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        trackingVisualizer = new TrackingVisualizer();
    });
} else {
    trackingVisualizer = new TrackingVisualizer();
}

console.log('Tracking UI loaded');
