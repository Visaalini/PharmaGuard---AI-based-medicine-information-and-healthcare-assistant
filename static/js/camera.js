document.addEventListener('DOMContentLoaded', function() {
    const cameraBtn = document.getElementById('camera-btn');
    const cameraSection = document.getElementById('camera-section');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image');

    let stream = null;

    cameraBtn.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            cameraSection.style.display = 'block';
            cameraBtn.style.display = 'none';
        } catch (error) {
            alert('Camera access denied or not available. Please upload an image instead.');
            console.error('Error accessing camera:', error);
        }
    });

    captureBtn.addEventListener('click', function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert canvas to blob
        canvas.toBlob(function(blob) {
            const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageInput.files = dataTransfer.files;

            // Hide video, show captured image
            video.style.display = 'none';
            canvas.style.display = 'block';
            captureBtn.style.display = 'none';
            retakeBtn.style.display = 'inline-block';
        });
    });

    retakeBtn.addEventListener('click', function() {
        video.style.display = 'block';
        canvas.style.display = 'none';
        captureBtn.style.display = 'inline-block';
        retakeBtn.style.display = 'none';
    });

    // Stop camera when form is submitted or page is unloaded
    uploadForm.addEventListener('submit', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    window.addEventListener('beforeunload', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
});
