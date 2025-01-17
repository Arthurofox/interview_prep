from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging
import cv2
import numpy as np
import base64
from typing import Dict
from src.core.config import get_settings
from src.processors.video import MediaPipeVideoProcessor
from src.processors.audio import SpeechAnalyzer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active analyzers
active_analyzers: Dict[str, tuple[MediaPipeVideoProcessor, SpeechAnalyzer]] = {}

@app.get("/")
async def get_html():
    """Serve the HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Interview Analysis</title>
            <style>
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .video-container {
                    position: relative;
                    margin-bottom: 20px;
                    width: 640px;
                    height: 480px;
                }
                #webcam {
                    position: absolute;
                    top: 0;
                    left: 0;
                }
                #overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    pointer-events: none;
                }
                .metrics {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .metric-box {
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    min-width: 150px;
                }
                .transcription {
                    width: 80%;
                    min-height: 100px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="video-container">
                    <video id="webcam" width="640" height="480" autoplay></video>
                    <canvas id="overlay" width="640" height="480"></canvas>
                </div>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Emotions</h3>
                        <div id="emotions"></div>
                    </div>
                    <div class="metric-box">
                        <h3>Audio Status</h3>
                        <div id="audioStatus"></div>
                    </div>
                </div>
                <button id="startBtn">Start Analysis</button>
                <button id="stopBtn" disabled>Stop Analysis</button>
                <div class="transcription">
                    <h3>Transcription</h3>
                    <div id="transcription"></div>
                </div>
            </div>
            <script>
                let ws;
                let mediaRecorder;
                let audioChunks = [];
                let videoInterval;
                
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const emotions = document.getElementById('emotions');
                const audioStatus = document.getElementById('audioStatus');
                const transcription = document.getElementById('transcription');
                
                async function setupMedia() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: true,
                            audio: true
                        });
                        
                        // Setup video
                        document.getElementById('webcam').srcObject = stream;
                        
                        // Setup audio recording
                        mediaRecorder = new MediaRecorder(stream);
                        mediaRecorder.ondataavailable = (event) => {
                            audioChunks.push(event.data);
                            // Send audio chunk to server
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                const audioBlob = new Blob([event.data], { type: 'audio/webm' });
                                audioBlob.arrayBuffer().then(buffer => {
                                    ws.send(JSON.stringify({
                                        type: 'audio',
                                        data: Array.from(new Uint8Array(buffer))
                                    }));
                                });
                            }
                        };
                        
                        return true;
                    } catch (error) {
                        console.error('Error accessing media devices:', error);
                        return false;
                    }
                }
                
                function startVideoStream() {
                    const video = document.getElementById('webcam');
                    const canvas = document.createElement('canvas');
                    canvas.width = video.width;
                    canvas.height = video.height;
                    const ctx = canvas.getContext('2d');
                    
                    videoInterval = setInterval(() => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            const frame = canvas.toDataURL('image/jpeg', 0.8);
                            ws.send(JSON.stringify({
                                type: 'video',
                                data: frame
                            }));
                        }
                    }, 50); // 20 FPS - adjust based on performance
                }
                
                async function connectWebSocket() {
                    ws = new WebSocket(`ws://${window.location.host}${window.location.pathname}ws`);
                    
                    ws.onopen = () => {
                        console.log('WebSocket connected');
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        
                        // Start sending video frames
                        startVideoStream();
                        
                        // Start audio recording
                        mediaRecorder.start(1000); // Collect audio in 1-second chunks
                        audioStatus.textContent = 'Recording...';
                    };
                    
                    ws.onclose = () => {
                        console.log('WebSocket disconnected');
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        clearInterval(videoInterval);
                        audioStatus.textContent = 'Stopped';
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'analysis') {
                            // Update emotions display
                            if (data.data.emotions) {
                                emotions.innerHTML = Object.entries(data.data.emotions)
                                    .map(([emotion, score]) => 
                                        `${emotion}: ${(score * 100).toFixed(1)}%`)
                                    .join('<br>');
                            }
                            
                            // Update overlay canvas with processed frame
                            if (data.data.processed_frame) {
                                const overlay = document.getElementById('overlay');
                                const ctx = overlay.getContext('2d');
                                const img = new Image();
                                img.onload = () => {
                                    ctx.clearRect(0, 0, overlay.width, overlay.height);
                                    ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
                                };
                                img.src = data.data.processed_frame;
                            }
                        } else if (data.type === 'transcription') {
                            // Update transcription
                            transcription.textContent = data.data;
                        } else if (data.type === 'error') {
                            console.error('Server error:', data.data);
                        }
                    };
                }
                
                startBtn.onclick = async () => {
                    if (await setupMedia()) {
                        connectWebSocket();
                    }
                };
                
                stopBtn.onclick = () => {
                    if (ws) {
                        if (mediaRecorder) {
                            mediaRecorder.stop();
                            mediaRecorder = null;
                        }
                        clearInterval(videoInterval);
                        ws.close();
                        audioStatus.textContent = 'Stopped';
                        ws = null;
                    }
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create unique session ID
    session_id = str(id(websocket))
    
    try:
        # Initialize analyzers
        video_processor = MediaPipeVideoProcessor()
        speech_analyzer = SpeechAnalyzer()
        active_analyzers[session_id] = (video_processor, speech_analyzer)
        
        logger.info(f"Session {session_id} initialized successfully")
        
        while True:
            try:
                data = await websocket.receive_json()
                if not data:  # Check if connection is closed
                    break
                
                if data["type"] == "video":
                    # Process video frame
                    frame_data = data["data"].split(",")[1]
                    frame_bytes = base64.b64decode(frame_data)
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    # Process frame with video processor
                    analysis_results = await video_processor.process_frame(frame)
                    
                    # Convert the processed frame (with drawings) back to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    processed_frame = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send results and processed frame back to client
                    await websocket.send_json({
                        "type": "analysis",
                        "data": {
                            "emotions": analysis_results.get("emotions", {}),
                            "faces_detected": analysis_results.get("faces_detected", 0),
                            "processed_frame": f"data:image/jpeg;base64,{processed_frame}"
                        }
                    })
                
                elif data["type"] == "audio":
                    # Process audio chunk
                    try:
                        raw_data = bytes(data["data"])
                        # Make sure the buffer size is a multiple of 4 (float32)
                        if len(raw_data) % 4 != 0:
                            padding = b'\x00' * (4 - (len(raw_data) % 4))
                            raw_data += padding
                        
                        # Analyze speech
                        speech_results = await speech_analyzer.analyze_chunk(raw_data)
                        if speech_results.get("transcription"):
                            await websocket.send_json({
                                "type": "transcription",
                                "data": speech_results["transcription"]
                            })
                    except Exception as e:
                        logger.error(f"Audio processing error: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "data": f"Audio processing error: {str(e)}"
                        })
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "data": str(e)
                })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Cleanup
        if session_id in active_analyzers:
            del active_analyzers[session_id]
            logger.info(f"Session {session_id} cleaned up")
        try:
            await websocket.close()
        except:
            pass
