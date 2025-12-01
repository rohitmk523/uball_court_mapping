/**
 * Court Renderer - Renders basketball court and tags on canvas
 */

class CourtRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error(`Canvas with id "${canvasId}" not found`);
            return;
        }

        this.ctx = this.canvas.getContext('2d');
        this.courtGeometry = null;
        this.tags = [];
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.margin = 50; // pixels

        this.loadCourtGeometry();
    }

    async loadCourtGeometry() {
        try {
            console.log('[courtRenderer] Loading court geometry...');
            const response = await fetch('/api/court/geometry');
            this.courtGeometry = await response.json();
            console.log('[courtRenderer] Court geometry loaded:', this.courtGeometry.bounds);
            this.setupCanvas();
            this.render();
        } catch (error) {
            console.error('Failed to load court geometry:', error);
        }
    }

    setupCanvas() {
        if (!this.courtGeometry) return;

        const bounds = this.courtGeometry.bounds;
        const courtWidth = bounds.width;
        const courtHeight = bounds.height;

        // Set canvas size to fit court with margins
        const maxWidth = this.canvas.parentElement.clientWidth - 40;
        const maxHeight = window.innerHeight * 0.8;

        // Calculate scale to fit canvas
        const scaleX = (maxWidth - 2 * this.margin) / courtWidth;
        const scaleY = (maxHeight - 2 * this.margin) / courtHeight;
        this.scale = Math.min(scaleX, scaleY);

        this.canvas.width = courtWidth * this.scale + 2 * this.margin;
        this.canvas.height = courtHeight * this.scale + 2 * this.margin;

        this.offsetX = this.margin - bounds.min_x * this.scale;
        this.offsetY = this.margin + bounds.max_y * this.scale; // Flip Y

        console.log('[courtRenderer] Canvas setup:', {
            canvasWidth: this.canvas.width,
            canvasHeight: this.canvas.height,
            courtBounds: bounds,
            scale: this.scale,
            offsetX: this.offsetX,
            offsetY: this.offsetY
        });
    }

    courtToCanvas(x, y) {
        // Convert court coordinates (cm) to canvas coordinates (pixels)
        const canvasX = x * this.scale + this.offsetX;
        const canvasY = -y * this.scale + this.offsetY; // Flip Y axis
        return { x: canvasX, y: canvasY };
    }

    render() {
        if (!this.courtGeometry) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Fill background
        this.ctx.fillStyle = '#f8f9fa';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw court elements
        this.drawPolylines();
        this.drawLines();
        this.drawCircles();
        this.drawTags();

        // Debug: Draw a big red test circle in the center
        if (this.tags && this.tags.length > 0) {
            const centerX = this.canvas.width / 2;
            const centerY = this.canvas.height / 2;
            this.ctx.fillStyle = 'red';
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, 20, 0, 2 * Math.PI);
            this.ctx.fill();
            console.log(`[courtRenderer] Drew test red circle at canvas center (${centerX}, ${centerY})`);
        }
    }

    drawPolylines() {
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;

        this.courtGeometry.polylines.forEach(polyline => {
            if (polyline.length < 2) return;

            this.ctx.beginPath();
            const start = this.courtToCanvas(polyline[0][0], polyline[0][1]);
            this.ctx.moveTo(start.x, start.y);

            for (let i = 1; i < polyline.length; i++) {
                const point = this.courtToCanvas(polyline[i][0], polyline[i][1]);
                this.ctx.lineTo(point.x, point.y);
            }

            this.ctx.closePath();
            this.ctx.stroke();
        });
    }

    drawLines() {
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;

        this.courtGeometry.lines.forEach(line => {
            const start = this.courtToCanvas(line[0][0], line[0][1]);
            const end = this.courtToCanvas(line[1][0], line[1][1]);

            this.ctx.beginPath();
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
            this.ctx.stroke();
        });
    }

    drawCircles() {
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;

        this.courtGeometry.circles.forEach(circle => {
            const center = this.courtToCanvas(circle[0][0], circle[0][1]);
            const radius = circle[1] * this.scale;

            this.ctx.beginPath();
            this.ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI);
            this.ctx.stroke();
        });
    }

    drawTags() {
        if (!this.tags || this.tags.length === 0) {
            return;
        }

        // Log first tag for debugging
        if (this.tags.length > 0) {
            const firstTag = this.tags[0];
            const pos = this.courtToCanvas(firstTag.x, firstTag.y);
            console.log(`[courtRenderer] Drawing tag ${firstTag.tag_id}: court (${firstTag.x}, ${firstTag.y}) -> canvas (${pos.x.toFixed(1)}, ${pos.y.toFixed(1)})`);
        }

        this.tags.forEach((tag, index) => {
            const pos = this.courtToCanvas(tag.x, tag.y);

            // Draw tag as colored circle
            const hue = (tag.tag_id * 137.5) % 360; // Golden angle for color distribution
            this.ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, 8, 0, 2 * Math.PI);
            this.ctx.fill();

            // Draw tag ID
            this.ctx.fillStyle = '#000000';
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(tag.tag_id, pos.x, pos.y - 12);
        });
    }

    updateTags(tags) {
        console.log(`[courtRenderer] updateTags called with ${tags.length} tags`, tags);
        this.tags = tags;
        this.render();
    }

    clear() {
        this.tags = [];
        this.render();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.courtRenderer = new CourtRenderer('courtCanvas');
        console.log('[courtRenderer] Initialized and set on window');
    });
} else {
    window.courtRenderer = new CourtRenderer('courtCanvas');
    console.log('[courtRenderer] Initialized and set on window');
}

console.log('Court Renderer loaded');
