class WebcamRecorder {
    constructor() {
        this.videoElement = null;
        this.stream = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
        this.currentVideoId = null;

        // For microphone selection
        this.availableMicrophones = [];
        this.selectedMicrophoneId = null;
    }

    async initialize() {
        try {
            console.log('Starting initialization...');
            
            // Cache key UI elements
            this.videoElement = document.getElementById('webcam');
            if (!this.videoElement) {
                throw new Error('Could not find video element');
            }

            // List available microphones and populate dropdown if found
            await this.listMicrophones();

            // Request access to camera and (optionally) the selected microphone
            const cameraSuccess = await this.requestCamera();
            if (!cameraSuccess) {
                throw new Error('Failed to initialize camera');
            }

            // Enable the start button after successful initialization
            const startBtn = document.getElementById('startBtn');
            if (startBtn) {
                startBtn.disabled = false;
            }

            console.log('Initialization completed successfully');
        } catch (error) {
            console.error('Initialization failed:', error);
            this.displayError('Initialization error: ' + error.message);
        }
    }

    /**
     * Lists available audio input devices (microphones) and populates a dropdown.
     */
    async listMicrophones() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.availableMicrophones = devices.filter(
                device => device.kind === 'audioinput'
            );

            if (this.availableMicrophones.length > 0) {
                const microphoneSelect = document.getElementById('microphoneSelect');
                if (microphoneSelect) {
                    // Clear existing options and add a default option
                    microphoneSelect.innerHTML = '';
                    const defaultOption = document.createElement('option');
                    defaultOption.value = '';
                    defaultOption.textContent = 'Select Microphone';
                    microphoneSelect.appendChild(defaultOption);

                    // Populate the select element with available microphones
                    this.availableMicrophones.forEach(mic => {
                        const option = document.createElement('option');
                        option.value = mic.deviceId;
                        option.textContent = mic.label || `Microphone ${mic.deviceId}`;
                        microphoneSelect.appendChild(option);
                    });

                    // Listen for microphone selection changes
                    microphoneSelect.addEventListener('change', event => {
                        this.selectedMicrophoneId = event.target.value;
                        console.log('Selected microphone:', this.selectedMicrophoneId);
                    });
                }
            } else {
                console.warn('No microphones found.');
            }
        } catch (error) {
            console.error('Error listing microphones:', error);
        }
    }

    async requestCamera() {
        console.log('Requesting camera/microphone access...');
        try {
            // Use the selected microphone if specified, otherwise allow any audio input
            const audioConstraints = this.selectedMicrophoneId
                ? { deviceId: { exact: this.selectedMicrophoneId } }
                : true;

            // Request media stream with desired video and audio constraints
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: audioConstraints,
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            if (!this.stream) {
                throw new Error('Failed to get camera/microphone stream');
            }

            // Attach the stream to the video element
            this.videoElement.srcObject = this.stream;

            // Wait for video metadata to load before playing
            await new Promise((resolve, reject) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play().then(resolve).catch(reject);
                };
                this.videoElement.onerror = reject;
            });

            return true;
        } catch (error) {
            console.error('Camera/microphone access error:', error);
            this.displayError(`Camera/microphone access error: ${error.message}. Please ensure your camera/mic is connected and permissions are granted.`);
            return false;
        }
    }

    startRecording() {
        try {
            // Reset recorded data
            this.recordedChunks = [];

            // Initialize MediaRecorder with proper mime type
            const mimeType = this.getSupportedMimeType();
            console.log(`Using MIME type: ${mimeType}`);
            
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 2500000 // 2.5 Mbps for better quality
            });

            this.mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.uploadRecording();
            };

            // Start recording with data available every 1 second
            this.mediaRecorder.start(1000);
            this.isRecording = true;
            this.startTime = Date.now();

            // Update UI state
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            const recordingStatus = document.getElementById('recordingStatus');
            recordingStatus.textContent = 'Recording...';
            recordingStatus.style.display = 'block';
            document.querySelector('.results-container').style.display = 'none';

            // Start the timer display
            this.startTimer();

            console.log('Started recording');
        } catch (error) {
            console.error('Error starting recording:', error);
            this.displayError('Failed to start recording: ' + error.message);
        }
    }
    
    getSupportedMimeType() {
        const types = [
            'video/webm;codecs=vp8,opus',
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=h264,opus',
            'video/webm',
            'video/mp4'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        // Default to webm if none supported (will likely fail, but we try)
        return 'video/webm';
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.stopTimer();

            // Update UI state
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            const recordingStatus = document.getElementById('recordingStatus');
            recordingStatus.textContent = 'Processing...';

            console.log('Stopped recording');
        }
    }

    startTimer() {
        const timerElement = document.getElementById('timer');
        this.startTime = Date.now();
        this.timerInterval = setInterval(() => {
            const elapsedTime = Date.now() - this.startTime;
            const minutes = Math.floor(elapsedTime / 60000);
            const seconds = Math.floor((elapsedTime % 60000) / 1000);
            timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    async uploadRecording() {
        try {
            // Create a blob from recorded chunks
            const blob = new Blob(this.recordedChunks, { 
                type: this.mediaRecorder.mimeType 
            });
            
            console.log(`Recording complete. Size: ${(blob.size / (1024 * 1024)).toFixed(2)} MB`);
            
            const formData = new FormData();
            formData.append('video', blob, 'interview.webm');

            // Update UI to show uploading state
            const recordingStatus = document.getElementById('recordingStatus');
            recordingStatus.textContent = 'Uploading...';

            // Upload the video to the server
            const response = await fetch('/upload-video', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error (${response.status}): ${errorText}`);
            }

            const result = await response.json();
            console.log('Upload successful:', result);
            this.currentVideoId = result.video_id;

            // Update UI and start polling for processing results
            recordingStatus.textContent = 'Upload complete! Processing video...';
            document.querySelector('.results-container').style.display = 'block';
            this.pollProcessingStatus(result.video_id);
        } catch (error) {
            console.error('Upload error:', error);
            this.displayError('Failed to upload video: ' + error.message);
            document.getElementById('recordingStatus').style.display = 'none';
        }
    }

    async pollProcessingStatus(videoId) {
        try {
            const response = await fetch(`/video-status/${videoId}`);
            if (!response.ok) {
                throw new Error(`Server error (${response.status})`);
            }
            
            const status = await response.json();

            if (status.status === "completed") {
                console.log("Processing complete, loading video...");

                // Update UI to show results and hide status message
                const resultsContainer = document.querySelector(".results-container");
                resultsContainer.style.display = "block";
                document.getElementById("recordingStatus").style.display = "none";

                // Format and display results
                document.getElementById("results").innerHTML = this.formatResults(status.results);

                // Load and play the processed video
                const processedVideo = document.getElementById("processedVideo");
                if (processedVideo) {
                    // Use direct video path or construct from video ID
                    let videoUrl;
                    const videoPath = status.results?.video_results?.analysis?.annotated_video_path;
                    
                    if (videoPath) {
                        // Extract filename from path and use as URL
                        const filename = videoPath.split('/').pop();
                        videoUrl = `/videos/${filename}`;
                    } else {
                        // Fallback to default path construction
                        videoUrl = `/video/${videoId}`;
                    }
                    
                    console.log("Loading video from:", videoUrl);
                    processedVideo.src = videoUrl;
                    processedVideo.style.display = "block";
                    processedVideo.load();
                    
                    // Only autoplay if user has interacted with the page
                    if (document.hasFocus()) {
                        processedVideo.play().catch(err => {
                            console.warn("Could not autoplay:", err);
                        });
                    }
                }
            } else if (status.status === "failed") {
                throw new Error(status.error || "Processing failed");
            } else {
                // Continue polling every 2 seconds
                setTimeout(() => this.pollProcessingStatus(videoId), 2000);
            }
        } catch (error) {
            console.error("Error checking status:", error);
            this.displayError("Error checking processing status: " + error.message);
        }
    }

    formatResults(results) {
        let html = '<h3>Analysis Results</h3>';

        const videoAnalysis = results?.video_results?.analysis;
        const audioAnalysis = results?.audio_results;

        if (videoAnalysis && videoAnalysis.visual_analysis) {
            const va = videoAnalysis.visual_analysis;
            html += '<div class="analysis-section">';
            html += '<h4>Visual Emotions (Video)</h4>';
            
            // Check if we have per-face data
            const faceKeys = Object.keys(va.average_emotions || {}).filter(k => k.startsWith('face_'));
            
            if (faceKeys.length > 0) {
                // We have multiple faces, show each one separately
                html += `<p>Detected ${va.face_count || faceKeys.length} faces in the video</p>`;
                
                for (const faceKey of faceKeys) {
                    const faceEmotions = va.average_emotions[faceKey];
                    const facePeakEmotions = va.peak_emotions[faceKey];
                    
                    html += `<h5>${faceKey.replace('_', ' ')}</h5>`;
                    html += '<div style="margin-left: 15px;">';
                    
                    // Average emotions for this face
                    html += '<strong>Average Emotions:</strong><ul class="emotion-list">';
                    for (const [emo, val] of Object.entries(faceEmotions)) {
                        html += `<li><strong>${emo}:</strong> ${(val * 100).toFixed(1)}%</li>`;
                    }
                    html += '</ul>';
                    
                    // Peak emotions for this face
                    if (facePeakEmotions) {
                        html += '<strong>Peak Emotions:</strong><ul class="emotion-list">';
                        for (const [emo, data] of Object.entries(facePeakEmotions)) {
                            html += `<li><strong>${emo}:</strong> ${(data.score * 100).toFixed(1)}% at ${data.timestamp.toFixed(1)}s</li>`;
                        }
                        html += '</ul>';
                    }
                    
                    html += '</div>';
                }
            } else if (va.average_emotions && Object.keys(va.average_emotions).length > 0) {
                // Single face or legacy format
                html += '<ul class="emotion-list">';
                for (const [emo, val] of Object.entries(va.average_emotions)) {
                    html += `<li><strong>${emo}:</strong> ${(val * 100).toFixed(1)}%</li>`;
                }
                html += '</ul>';
                
                if (va.peak_emotions) {
                    html += '<h5>Peak Emotions</h5>';
                    html += '<ul class="emotion-list">';
                    for (const [emo, data] of Object.entries(va.peak_emotions)) {
                        html += `<li><strong>${emo}:</strong> ${(data.score * 100).toFixed(1)}% at ${data.timestamp.toFixed(1)}s</li>`;
                    }
                    html += '</ul>';
                }
            }
            
            html += '</div>';
        }

        if (audioAnalysis?.transcription || audioAnalysis?.emotions) {
            html += '<div class="analysis-section">';
            html += '<h4>Audio Analysis</h4>';

            if (audioAnalysis.transcription) {
                html += `<p><strong>Transcription:</strong> ${audioAnalysis.transcription}</p>`;
            }
            
            if (audioAnalysis.emotions) {
                const avg = audioAnalysis.emotions.average_emotions;
                const peak = audioAnalysis.emotions.peak_emotions;
                
                if (avg && Object.keys(avg).length > 0) {
                    html += '<h5>Average Audio Emotions</h5><ul>';
                    for (const [emotion, score] of Object.entries(avg)) {
                        html += `<li>${emotion}: ${(score * 100).toFixed(1)}%</li>`;
                    }
                    html += '</ul>';
                }
                
                if (peak && Object.keys(peak).length > 0) {
                    html += '<h5>Peak Audio Emotions</h5><ul>';
                    for (const [emotion, data] of Object.entries(peak)) {
                        html += `<li>${emotion}: ${(data.score * 100).toFixed(1)}% (timestamp: ${data.timestamp.toFixed(1)}s)</li>`;
                    }
                    html += '</ul>';
                }
            }
            
            html += '</div>';
        }

        const info = results?.video_results?.video_info;
        if (info) {
            html += '<div class="analysis-section">';
            html += '<h4>Video Information</h4>';
            html += `<ul>
                        <li>Duration: ${Math.round(info.duration)} s</li>
                        <li>Total Frames: ${info.total_frames}</li>
                        <li>FPS: ${info.fps.toFixed(1)}</li>
                        <li>Resolution: ${info.width}×${info.height}</li>
                    </ul>`;
            html += '</div>';
        }

        return html;
    }

    displayError(message) {
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
        console.error(message);
    }
}

// Initialize the recorder when the page loads
window.addEventListener('DOMContentLoaded', async () => {
    console.log('Page loaded, initializing recorder...');
    const recorder = new WebcamRecorder();

    try {
        await recorder.initialize();

        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (startBtn) {
            startBtn.addEventListener('click', () => {
                recorder.startRecording();
            });
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                recorder.stopRecording();
            });
        }
        
        // Add keyboard shortcuts for convenience
        document.addEventListener('keydown', (event) => {
            // Space bar to start/stop recording (if not in an input field)
            if (event.code === 'Space' && document.activeElement.tagName !== 'INPUT' && 
                document.activeElement.tagName !== 'TEXTAREA') {
                event.preventDefault();
                
                if (!recorder.isRecording && !startBtn.disabled) {
                    recorder.startRecording();
                } else if (recorder.isRecording && !stopBtn.disabled) {
                    recorder.stopRecording();
                }
            }
            
            // Escape key to stop recording
            if (event.code === 'Escape' && recorder.isRecording) {
                recorder.stopRecording();
            }
        });
    } catch (error) {
        console.error('Failed to initialize recorder:', error);
        recorder.displayError('Failed to initialize: ' + error.message);
    }
});