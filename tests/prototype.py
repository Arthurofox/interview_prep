from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from datetime import datetime
import asyncio
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

@app.get("/")
async def get_html():
    """Serve a simple HTML page with webcam display"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Interview Prep Prototype</title>
            <style>
                .container { 
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                }
                .video-container {
                    margin-bottom: 20px;
                }
                #status {
                    margin-top: 10px;
                    font-family: Arial, sans-serif;
                }
                #transcription {
                    margin-top: 10px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    width: 80%;
                    min-height: 100px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="video-container">
                    <video id="webcam" width="640" height="480" autoplay></video>
                    <canvas id="overlay" width="640" height="480" style="display: none;"></canvas>
                </div>
                <button id="startBtn">Start Session</button>
                <button id="stopBtn" disabled>Stop Session</button>
                <div id="status">Not connected</div>
                <div id="transcription"></div>
            </div>
            <script>
                let ws;
                let mediaRecorder;
                let audioChunks = [];
                
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const status = document.getElementById('status');
                const transcription = document.getElementById('transcription');
                
                async function setupWebcam() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            video: true,
                            audio: true 
                        });
                        document.getElementById('webcam').srcObject = stream;
                        
                        // Set up audio recording
                        mediaRecorder = new MediaRecorder(stream);
                        mediaRecorder.ondataavailable = (event) => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            // Here you would send the audio data to the server
                            audioChunks = [];
                        };
                        
                        return true;
                    } catch (error) {
                        console.error('Error accessing webcam:', error);
                        status.textContent = 'Error accessing webcam';
                        return false;
                    }
                }
                
                async function connectWebSocket() {
                    ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onopen = () => {
                        status.textContent = 'Connected';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        startVideoStream();
                        mediaRecorder.start(1000); // Collect audio in 1-second chunks
                    };
                    
                    ws.onclose = () => {
                        status.textContent = 'Disconnected';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.type === 'transcription') {
                            transcription.textContent = data.text;
                        }
                    };
                }
                
                function startVideoStream() {
                    const video = document.getElementById('webcam');
                    const canvas = document.createElement('canvas');
                    canvas.width = video.width;
                    canvas.height = video.height;
                    const ctx = canvas.getContext('2d');
                    
                    setInterval(() => {
                        if (ws.readyState === WebSocket.OPEN) {
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            const frame = canvas.toDataURL('image/jpeg', 0.5);
                            ws.send(JSON.stringify({
                                type: 'video_frame',
                                data: frame
                            }));
                        }
                    }, 100); // Send 10 frames per second
                }
                
                startBtn.onclick = async () => {
                    if (await setupWebcam()) {
                        connectWebSocket();
                    }
                };
                
                stopBtn.onclick = () => {
                    if (ws) {
                        ws.close();
                        mediaRecorder.stop();
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
    
    # Generate a unique ID for this connection
    connection_id = str(datetime.now().timestamp())
    active_connections[connection_id] = websocket
    
    try:
        # Initialize face detection
        with mp_face_detection.FaceDetection(
            model_selection=0,  # Model for close-range face detection
            min_detection_confidence=0.5
        ) as face_detection:
            
            while True:
                try:
                    # Receive and process the frame
                    data = await websocket.receive_json()
                    
                    if data["type"] == "video_frame":
                        # Process the video frame
                        frame_data = data["data"].split(",")[1]  # Remove data URL prefix
                        frame_bytes = base64.b64decode(frame_data)
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        # Convert to RGB for MediaPipe
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(frame_rgb)
                        
                        # Process face detection results
                        if results.detections:
                            # For prototype, just count faces and send basic info
                            face_data = {
                                "type": "face_detection",
                                "faces_found": len(results.detections),
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket.send_json(face_data)
                            
                            # In a full implementation, you would:
                            # 1. Extract face landmarks
                            # 2. Analyze emotions
                            # 3. Track face movement
                            # 4. Generate detailed metrics
                    
                    # Here you would also handle audio data and perform:
                    # 1. Speech-to-text conversion
                    # 2. Natural language processing
                    # 3. AI response generation
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                
    except Exception as e:
        logger.exception("WebSocket error")
    finally:
        del active_connections[connection_id]
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)