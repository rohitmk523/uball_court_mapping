/**
 * Tag Animator - Animates tag movement with FPS control
 */

class TagAnimator {
    constructor() {
        this.tagData = {}; // Map of tag_id -> array of positions
        this.tagIds = [];
        this.timeRange = [0, 0];
        this.currentTimestamp = 0;
        this.isPlaying = false;
        this.targetFPS = 30;
        this.actualFPS = 0;
        this.lastFrameTime = 0;
        this.animationFrameId = null;
        this.fpsCounter = 0;
        this.fpsLastUpdate = 0;

        this.initializeUI();
    }

    initializeUI() {
        // Process button
        document.getElementById('processBtn')?.addEventListener('click', () => this.processLogFile());

        // Playback controls
        document.getElementById('playBtn')?.addEventListener('click', () => this.play());
        document.getElementById('pauseBtn')?.addEventListener('click', () => this.pause());
        document.getElementById('resetBtn')?.addEventListener('click', () => this.reset());

        // FPS slider
        const fpsSlider = document.getElementById('fpsSlider');
        if (fpsSlider) {
            fpsSlider.addEventListener('input', (e) => {
                this.targetFPS = parseInt(e.target.value);
                document.getElementById('fpsValue').textContent = this.targetFPS;
            });
        }

        // Timeline slider
        const timeline = document.getElementById('timeline');
        if (timeline) {
            timeline.addEventListener('input', (e) => {
                const progress = parseFloat(e.target.value) / 100;
                this.currentTimestamp = this.timeRange[0] + progress * (this.timeRange[1] - this.timeRange[0]);
                this.updateFrame();
            });
        }
    }

    async processLogFile() {
        const btn = document.getElementById('processBtn');
        const status = document.getElementById('processingStatus');

        btn.disabled = true;
        status.textContent = 'Processing log file...';
        status.className = 'status info';

        try {
            // Process log file
            const response = await fetch('/api/tags/process', {
                method: 'POST'
            });
            const result = await response.json();

            if (result.status !== 'complete') {
                throw new Error(result.message || 'Processing failed');
            }

            status.textContent = result.message;
            status.className = 'status success';

            // Load tag data
            await this.loadTagData();

            // Enable controls
            document.getElementById('playBtn').disabled = false;
            document.getElementById('pauseBtn').disabled = false;
            document.getElementById('resetBtn').disabled = false;
            document.getElementById('fpsSlider').disabled = false;
            document.getElementById('timeline').disabled = false;

        } catch (error) {
            status.textContent = 'Error: ' + error.message;
            status.className = 'status error';
            btn.disabled = false;
        }
    }

    async loadTagData() {
        try {
            // Get list of tag IDs
            const listResponse = await fetch('/api/tags/list');
            this.tagIds = await listResponse.json();

            document.getElementById('activeTags').textContent = this.tagIds.length;

            // Load data for each tag
            const tagListDiv = document.getElementById('tagList');
            tagListDiv.innerHTML = '';

            for (const tagId of this.tagIds) {
                const response = await fetch(`/api/tags/${tagId}`);
                const data = await response.json();
                this.tagData[tagId] = data.positions;

                // Add to tag list
                const tagItem = document.createElement('div');
                tagItem.className = 'tag-item';
                tagItem.textContent = `Tag ${tagId}: ${data.positions.length} positions`;
                tagListDiv.appendChild(tagItem);
            }

            // Calculate overall time range
            let minTs = Infinity;
            let maxTs = -Infinity;

            for (const positions of Object.values(this.tagData)) {
                if (positions.length > 0) {
                    minTs = Math.min(minTs, positions[0].timestamp);
                    maxTs = Math.max(maxTs, positions[positions.length - 1].timestamp);
                }
            }

            this.timeRange = [minTs, maxTs];
            this.currentTimestamp = minTs;

            console.log(`Loaded ${this.tagIds.length} tags, time range: ${minTs} - ${maxTs}`);
            console.log(`Duration: ${((maxTs - minTs) / 1000000).toFixed(1)} seconds`);

            // Render initial frame
            this.updateFrame();

        } catch (error) {
            console.error('Failed to load tag data:', error);
        }
    }

    play() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        this.lastFrameTime = performance.now();
        this.fpsLastUpdate = performance.now();
        this.fpsCounter = 0;
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
        this.currentTimestamp = this.timeRange[0];
        this.updateFrame();
        document.getElementById('timeline').value = 0;
    }

    animate() {
        if (!this.isPlaying) return;

        const now = performance.now();
        const deltaTime = now - this.lastFrameTime;
        const frameInterval = 1000 / this.targetFPS;

        if (deltaTime >= frameInterval) {
            // Advance timestamp
            const timeStep = (this.timeRange[1] - this.timeRange[0]) / (this.targetFPS * 10); // 10 seconds to play full range
            this.currentTimestamp += timeStep;

            // Loop if reached end
            if (this.currentTimestamp > this.timeRange[1]) {
                this.currentTimestamp = this.timeRange[0];
            }

            this.updateFrame();

            // Update timeline slider
            const progress = (this.currentTimestamp - this.timeRange[0]) / (this.timeRange[1] - this.timeRange[0]);
            document.getElementById('timeline').value = progress * 100;

            // Calculate FPS
            this.fpsCounter++;
            if (now - this.fpsLastUpdate >= 1000) {
                this.actualFPS = this.fpsCounter;
                document.getElementById('currentFps').textContent = this.actualFPS;
                this.fpsCounter = 0;
                this.fpsLastUpdate = now;
            }

            this.lastFrameTime = now - (deltaTime % frameInterval);
        }

        this.animationFrameId = requestAnimationFrame(() => this.animate());
    }

    updateFrame() {
        // Get tag positions at current timestamp
        const currentTags = [];

        for (const tagId of this.tagIds) {
            const positions = this.tagData[tagId];
            if (!positions || positions.length === 0) continue;

            // Use binary search to find closest position
            const closest = this.findClosestPosition(positions, this.currentTimestamp);

            if (closest) {
                currentTags.push({
                    tag_id: tagId,
                    x: closest.x,
                    y: closest.y,
                    timestamp: closest.timestamp
                });
            }
        }

        console.log(`[updateFrame] Current timestamp: ${this.currentTimestamp}, Tags found: ${currentTags.length}`, currentTags);

        // Update court renderer
        if (window.courtRenderer) {
            window.courtRenderer.updateTags(currentTags);
        } else {
            console.error('[tagAnimator] window.courtRenderer not found!');
        }

        // Update timestamp display
        document.getElementById('currentTimestamp').textContent = this.currentTimestamp.toFixed(0);
    }

    findClosestPosition(positions, timestamp) {
        if (!positions || positions.length === 0) return null;

        // Binary search for closest timestamp
        let left = 0;
        let right = positions.length - 1;

        // Handle edge cases
        if (timestamp <= positions[0].timestamp) return positions[0];
        if (timestamp >= positions[right].timestamp) return positions[right];

        while (left <= right) {
            const mid = Math.floor((left + right) / 2);

            if (positions[mid].timestamp === timestamp) {
                return positions[mid];
            }

            if (positions[mid].timestamp < timestamp) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // At this point, right < left
        // positions[right].timestamp < timestamp < positions[left].timestamp
        // Return the closer one
        const diffLeft = Math.abs(positions[left].timestamp - timestamp);
        const diffRight = Math.abs(positions[right].timestamp - timestamp);

        return diffLeft < diffRight ? positions[left] : positions[right];
    }
}

// Global instance
let tagAnimator = null;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        tagAnimator = new TagAnimator();
    });
} else {
    tagAnimator = new TagAnimator();
}

console.log('Tag Animator loaded');
