/**
 * PharmaGuard Camera Handler
 * Handles device camera access with comprehensive error handling and browser support
 */

class CameraHandler {
    constructor(config = {}) {
        this.config = {
            videoElementId: config.videoElementId || 'video',
            canvasElementId: config.canvasElementId || 'canvas',
            cameraSectionId: config.cameraSectionId || 'camera-section',
            uploadPlaceholderId: config.uploadPlaceholderId || 'upload-placeholder',
            previewContainerId: config.previewContainerId || 'preview-container',
            imageInputId: config.imageInputId || 'image',
            imagePreviewId: config.imagePreviewId || 'image-preview',
            ...config
        };

        this.elements = {};
        this.stream = null;
        this.isCameraActive = false;
        this.isSupported = this.checkSupport();

        this.initElements();
        this.setupErrorContainer();
    }

    /**
     * Check browser support for getUserMedia API
     */
    checkSupport() {
        const getUserMedia =
            navigator.mediaDevices?.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia ||
            navigator.msGetUserMedia;

        const isHTTPS = window.location.protocol === 'https:' || window.location.hostname === 'localhost';
        const isSupported = !!getUserMedia;

        return {
            isSupported,
            getUserMedia: navigator.mediaDevices?.getUserMedia,
            isHTTPS,
            isChrome: /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor),
            isEdge: /Edg/.test(navigator.userAgent),
            isFirefox: /Firefox/.test(navigator.userAgent),
            isSafari: /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent)
        };
    }

    /**
     * Initialize DOM elements
     */
    initElements() {
        try {
            this.elements = {
                video: document.getElementById(this.config.videoElementId),
                canvas: document.getElementById(this.config.canvasElementId),
                cameraSection: document.getElementById(this.config.cameraSectionId),
                uploadPlaceholder: document.getElementById(this.config.uploadPlaceholderId),
                previewContainer: document.getElementById(this.config.previewContainerId),
                imageInput: document.getElementById(this.config.imageInputId),
                imagePreview: document.getElementById(this.config.imagePreviewId)
            };

            // Verify all required elements exist
            const missingElements = Object.entries(this.elements)
                .filter(([_, elem]) => !elem)
                .map(([key]) => key);

            if (missingElements.length > 0) {
                console.warn(`Missing elements: ${missingElements.join(', ')}`);
            }
        } catch (error) {
            console.error('Error initializing elements:', error);
        }
    }

    /**
     * Setup error notification container
     */
    setupErrorContainer() {
        if (!this.elements.cameraSection) return;

        let errorContainer = document.getElementById('camera-error-container');
        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.id = 'camera-error-container';
            this.elements.cameraSection.parentNode.insertBefore(errorContainer, this.elements.cameraSection);
        }
    }

    /**
     * Show error message with specific styling
     */
    showError(title, message, errorType = 'permission') {
        const errorContainer = document.getElementById('camera-error-container');
        if (!errorContainer) return;

        const iconClass = this.getErrorIcon(errorType);

        errorContainer.innerHTML = `
            <div class="camera-error-alert" style="
                background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
                border: 2px solid #ff6b6b;
                border-left: 4px solid #ff6b6b;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                display: flex;
                gap: 15px;
                align-items: flex-start;
            ">
                <div style="font-size: 1.5rem; color: #ff6b6b; flex-shrink: 0;">
                    <i class="fas ${iconClass}"></i>
                </div>
                <div style="flex-grow: 1;">
                    <h5 style="margin: 0 0 8px 0; color: #c92a2a; font-weight: 600;">
                        ${title}
                    </h5>
                    <p style="margin: 0; color: #666; font-size: 0.95rem; line-height: 1.5;">
                        ${message}
                    </p>
                    <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                        <button type="button" class="btn-error-action" data-action="retry" style="
                            padding: 6px 14px;
                            background: #ff6b6b;
                            color: white;
                            border: none;
                            border-radius: 6px;
                            cursor: pointer;
                            font-size: 0.85rem;
                            font-weight: 500;
                            transition: all 0.3s ease;
                        ">
                            <i class="fas fa-redo me-1"></i>Try Again
                        </button>
                        <button type="button" class="btn-error-action" data-action="cancel" style="
                            padding: 6px 14px;
                            background: transparent;
                            color: #666;
                            border: 1px solid #ccc;
                            border-radius: 6px;
                            cursor: pointer;
                            font-size: 0.85rem;
                            font-weight: 500;
                            transition: all 0.3s ease;
                        ">
                            <i class="fas fa-times me-1"></i>Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Attach event listeners
        const retryBtn = errorContainer.querySelector('[data-action="retry"]');
        const cancelBtn = errorContainer.querySelector('[data-action="cancel"]');

        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.startCamera());
        }
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.closeCamera());
        }

        this.hideElement(this.elements.cameraSection);
    }

    /**
     * Get appropriate icon for error type
     */
    getErrorIcon(errorType) {
        const icons = {
            permission: 'fa-lock',
            notfound: 'fa-video',
            https: 'fa-shield-alt',
            notallowed: 'fa-ban',
            overconstrained: 'fa-sliders-h',
            abort: 'fa-exclamation-circle',
            unknown: 'fa-exclamation-triangle'
        };
        return icons[errorType] || icons.unknown;
    }

    /**
     * Clear error container
     */
    clearError() {
        const errorContainer = document.getElementById('camera-error-container');
        if (errorContainer) {
            errorContainer.innerHTML = '';
        }
    }

    /**
     * Start camera with comprehensive error handling
     */
    async startCamera() {
        // Check HTTPS requirement
        if (!this.isSupported.isHTTPS) {
            this.showError(
                'HTTPS Required',
                'Camera access requires HTTPS. Please use a secure connection. Localhost is supported for development.',
                'https'
            );
            return;
        }

        // Check browser support
        if (!this.isSupported.isSupported) {
            this.showError(
                'Camera Not Supported',
                `Your browser (${this.getBrowserName()}) doesn't support camera access. Please use Chrome, Edge, Firefox, or Safari.`,
                'notfound'
            );
            return;
        }

        this.clearError();
        this.showElement(this.elements.cameraSection);
        this.hideElement(this.elements.uploadPlaceholder);
        this.hideElement(this.elements.previewContainer);

        try {
            // Check camera permission status first (if available)
            const permissionStatus = await this.checkCameraPermission();
            if (permissionStatus === 'denied') {
                this.showError(
                    'Camera Permission Denied',
                    'You have blocked camera access. Please enable camera permissions in your browser settings and try again.',
                    'permission'
                );
                return;
            }

            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            if (this.elements.video) {
                this.elements.video.srcObject = this.stream;
                this.isCameraActive = true;

                // Handle video play errors
                this.elements.video.onloadedmetadata = () => {
                    this.elements.video.play().catch(err => {
                        this.showError(
                            'Video Playback Error',
                            'Failed to start video playback. Please try again.',
                            'abort'
                        );
                        console.error('Video play error:', err);
                    });
                };
            }

        } catch (error) {
            this.handleCameraError(error);
        }
    }

    /**
     * Check camera permission status using Permissions API
     */
    async checkCameraPermission() {
        try {
            if (!navigator.permissions) {
                return 'unknown';
            }
            const permission = await navigator.permissions.query({ name: 'camera' });
            return permission.state; // 'granted', 'denied', 'prompt'
        } catch (error) {
            return 'unknown';
        }
    }

    /**
     * Handle various camera errors
     */
    handleCameraError(error) {
        console.error('Camera error:', error);

        let title = 'Camera Error';
        let message = 'Unable to access the camera. Please try again.';
        let errorType = 'unknown';

        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            title = 'Camera Permission Denied';
            message = 'You need to allow camera access. Go to your browser settings to enable camera permissions.';
            errorType = 'permission';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            title = 'No Camera Detected';
            message = 'No camera device was found on your system. Please connect a camera and try again.';
            errorType = 'notfound';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            title = 'Camera Busy or Unavailable';
            message = 'The camera is busy or unavailable. Close other apps using the camera and try again.';
            errorType = 'notallowed';
        } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
            title = 'Camera Constraints Not Met';
            message = 'Your camera doesn\'t support the required settings. Try with different camera settings.';
            errorType = 'overconstrained';
        } else if (error.name === 'AbortError') {
            title = 'Camera Access Aborted';
            message = 'Camera access was aborted. Please try again.';
            errorType = 'abort';
        } else if (error.name === 'SecurityError') {
            title = 'Security Error';
            message = 'Camera access is blocked for security reasons. Please use HTTPS and try again.';
            errorType = 'https';
        }

        this.showError(title, message, errorType);
        this.isCameraActive = false;
    }

    /**
     * Capture image from video stream
     */
    captureImage(callback) {
        if (!this.elements.video || !this.elements.canvas) {
            console.error('Video or canvas element not found');
            return;
        }

        try {
            this.elements.canvas.width = this.elements.video.videoWidth;
            this.elements.canvas.height = this.elements.video.videoHeight;

            const ctx = this.elements.canvas.getContext('2d');
            if (!ctx) {
                console.error('Could not get canvas context');
                return;
            }

            ctx.drawImage(this.elements.video, 0, 0);

            this.elements.canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                    if (callback) callback(file);
                } else {
                    console.error('Canvas to blob conversion failed');
                }
            }, 'image/jpeg', 0.95);

        } catch (error) {
            console.error('Error capturing image:', error);
        }
    }

    /**
     * Stop camera stream
     */
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
            });
            this.stream = null;
        }
        this.isCameraActive = false;
    }

    /**
     * Close camera section
     */
    closeCamera() {
        this.stopCamera();
        this.clearError();
        this.hideElement(this.elements.cameraSection);
        this.showElement(this.elements.uploadPlaceholder);
    }

    /**
     * Utility: hide element
     */
    hideElement(elem) {
        if (elem) elem.classList.add('is-hidden');
    }

    /**
     * Utility: show element
     */
    showElement(elem) {
        if (elem) elem.classList.remove('is-hidden');
    }

    /**
     * Get browser name
     */
    getBrowserName() {
        if (this.isSupported.isChrome) return 'Chrome';
        if (this.isSupported.isEdge) return 'Edge';
        if (this.isSupported.isFirefox) return 'Firefox';
        if (this.isSupported.isSafari) return 'Safari';
        return 'your browser';
    }

    /**
     * Get camera support info
     */
    getSupportInfo() {
        return {
            isSupported: this.isSupported.isSupported,
            isHTTPS: this.isSupported.isHTTPS,
            browser: this.getBrowserName(),
            browserFeatures: {
                chrome: this.isSupported.isChrome,
                edge: this.isSupported.isEdge,
                firefox: this.isSupported.isFirefox,
                safari: this.isSupported.isSafari
            }
        };
    }

    /**
     * Cleanup on page unload
     */
    cleanup() {
        this.stopCamera();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.cameraHandler = new CameraHandler();
});
