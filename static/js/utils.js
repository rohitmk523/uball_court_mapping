/**
 * Utility functions for UI interactions and feedback
 */

class UIUtils {
    /**
     * Show a loading overlay with message
     */
    static showLoadingOverlay(message = 'Processing...', subtext = '') {
        // Remove existing overlay if present
        this.hideLoadingOverlay();

        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = 'loadingOverlay';

        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';

        const text = document.createElement('div');
        text.className = 'loading-text';
        text.textContent = message;

        overlay.appendChild(spinner);
        overlay.appendChild(text);

        if (subtext) {
            const subtextEl = document.createElement('div');
            subtextEl.className = 'loading-subtext';
            subtextEl.textContent = subtext;
            overlay.appendChild(subtextEl);
        }

        document.body.appendChild(overlay);
        return overlay;
    }

    /**
     * Hide loading overlay
     */
    static hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    /**
     * Show inline spinner next to element
     */
    static showInlineSpinner(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return null;

        const spinner = document.createElement('span');
        spinner.className = 'loading-spinner';
        spinner.id = `spinner-${elementId}`;
        element.appendChild(spinner);
        return spinner;
    }

    /**
     * Hide inline spinner
     */
    static hideInlineSpinner(elementId) {
        const spinner = document.getElementById(`spinner-${elementId}`);
        if (spinner) {
            spinner.remove();
        }
    }

    /**
     * Show status message
     */
    static showStatus(elementId, message, type = 'info') {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.textContent = message;
        element.className = `status ${type}`;
        element.style.display = 'block';
    }

    /**
     * Hide status message
     */
    static hideStatus(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    }

    /**
     * Create and show a badge
     */
    static createBadge(text, type = 'info') {
        const badge = document.createElement('span');
        badge.className = `badge ${type}`;
        badge.textContent = text;
        return badge;
    }

    /**
     * Format timestamp for display
     */
    static formatTimestamp(timestamp) {
        return timestamp.toLocaleString();
    }

    /**
     * Format number with commas
     */
    static formatNumber(num) {
        return num.toLocaleString();
    }

    /**
     * Validate number input
     */
    static validateNumberInput(value, min, max) {
        const num = parseInt(value);
        if (isNaN(num)) return { valid: false, message: 'Please enter a valid number' };
        if (min !== undefined && num < min) return { valid: false, message: `Value must be at least ${min}` };
        if (max !== undefined && num > max) return { valid: false, message: `Value must be at most ${max}` };
        return { valid: true };
    }

    /**
     * Debounce function calls
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Handle API errors consistently
     */
    static handleAPIError(error, statusElementId) {
        console.error('API Error:', error);

        let message = 'An error occurred. Please try again.';
        if (error.message) {
            message = error.message;
        } else if (typeof error === 'string') {
            message = error;
        }

        if (statusElementId) {
            this.showStatus(statusElementId, message, 'error');
        }

        return message;
    }

    /**
     * Fetch with timeout and error handling
     */
    static async fetchWithTimeout(url, options = {}, timeout = 30000) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }

            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timed out. Please try again.');
            }
            throw error;
        }
    }

    /**
     * Add tooltip to element
     */
    static addTooltip(elementId, tooltipText) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const wrapper = document.createElement('span');
        wrapper.className = 'tooltip';

        const tooltip = document.createElement('span');
        tooltip.className = 'tooltiptext';
        tooltip.textContent = tooltipText;

        element.parentNode.insertBefore(wrapper, element);
        wrapper.appendChild(element);
        wrapper.appendChild(tooltip);
    }

    /**
     * Disable element with visual feedback
     */
    static disableElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;

        if (element.tagName === 'BUTTON' || element.tagName === 'INPUT') {
            element.disabled = true;
        } else {
            element.classList.add('disabled-overlay');
        }
    }

    /**
     * Enable element
     */
    static enableElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;

        if (element.tagName === 'BUTTON' || element.tagName === 'INPUT') {
            element.disabled = false;
        } else {
            element.classList.remove('disabled-overlay');
        }
    }

    /**
     * Create progress bar
     */
    static createProgressBar(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        const progressContainer = document.createElement('div');
        progressContainer.className = 'progress-bar-container';
        progressContainer.id = `progress-${containerId}`;

        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.style.width = '0%';

        progressContainer.appendChild(progressBar);
        container.appendChild(progressContainer);

        return {
            update: (percent) => {
                progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
            },
            remove: () => {
                progressContainer.remove();
            }
        };
    }

    /**
     * Animate canvas highlight
     */
    static highlightCanvas(canvasId, duration = 1000) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        canvas.classList.add('canvas-highlight');
        setTimeout(() => {
            canvas.classList.remove('canvas-highlight');
        }, duration);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIUtils;
}
