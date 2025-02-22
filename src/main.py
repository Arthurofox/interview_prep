from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
from typing import Dict
from datetime import datetime
from pathlib import Path
import uuid
import aiofiles

from .config import get_settings
from .processors.video import VideoProcessor  
from .processors.audio import AudioProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories for processed videos and uploads
PROCESSED_VIDEOS_DIR = Path("uploads")
PROCESSED_VIDEOS_DIR.mkdir(exist_ok=True)

# Get the absolute path to the project root and static directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "interface" / "static"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Get settings
settings = get_settings()

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG
)

# Configure CORS to allow all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists and mount static files
if not STATIC_DIR.exists():
    logger.error(f"Static directory not found at {STATIC_DIR}")
    os.makedirs(STATIC_DIR, exist_ok=True)
    logger.info(f"Created static directory at {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Mount processed videos at /videos/
app.mount("/videos", StaticFiles(directory=str(PROCESSED_VIDEOS_DIR)), name="videos")

# Initialize processors
# Detect appropriate device (CUDA, MPS, or CPU)
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        logger.info("Using CPU for processing")
except Exception as e:
    logger.error(f"Error detecting device: {e}")
    device = "cpu"
    logger.info("Falling back to CPU for processing")

# Initialize processors with detected device
model_path = "best_emotion_model.pth"
if os.path.exists(model_path):
    video_processor = VideoProcessor(model_path=model_path, device=device)
    logger.info(f"Initialized VideoProcessor with model: {model_path}")
else:
    logger.warning(f"Model file not found: {model_path}")
    video_processor = VideoProcessor(device=device)
    logger.info("Initialized VideoProcessor without emotion model")

audio_processor = AudioProcessor(device="cpu")  

# In-memory dictionary to track processing status and results
processing_tasks: Dict[str, dict] = {}

@app.get("/")
async def get_html():
    """
    Serve a simple HTML interface that uses JS to record from the webcam
    and upload the video to this API.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Interview Recorder</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="/static/js/webcam-recorder.js" defer></script>
            <style>
                body {
                    margin: 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #f4f4f4;
                    color: #333;
                }
                .container {
                    max-width: 1280px;
                    margin: 0 auto;
                    padding: 20px;
                    text-align: center;
                }
                h1 {
                    margin-bottom: 20px;
                }
                .video-container {
                    position: relative;
                    margin: 0 auto 20px;
                    width: 100%;
                    max-width: 1280px;
                    height: 0;
                    padding-bottom: 56.25%;
                    border: 2px solid #ccc;
                    background: #000;
                    border-radius: 8px;
                    overflow: hidden;
                }
                #webcam {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                #recordingStatus {
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(0, 0, 0, 0.75);
                    color: #fff;
                    padding: 5px 10px;
                    border-radius: 4px;
                    display: none;
                    font-size: 14px;
                }
                .controls {
                    margin: 20px 0;
                }
                button {
                    padding: 12px 24px;
                    margin: 0 10px;
                    font-size: 16px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                button:disabled {
                    background-color: #cccccc;
                    cursor: not-allowed;
                }
                #startBtn {
                    background-color: #4CAF50;
                    color: #fff;
                }
                #startBtn:hover:not(:disabled) {
                    background-color: #45a049;
                }
                #stopBtn {
                    background-color: #f44336;
                    color: #fff;
                }
                #stopBtn:hover:not(:disabled) {
                    background-color: #e53935;
                }
                .timer {
                    font-size: 24px;
                    margin: 10px 0;
                }
                #error-message {
                    background: #ffdddd;
                    border: 1px solid #f44336;
                    color: #a94442;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    display: none;
                }
                .results-container {
                    margin-top: 20px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    background: #fff;
                    display: none;
                }
                .split-view {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                @media (min-width: 768px) {
                    .split-view {
                        flex-direction: row;
                    }
                }
                .video-player, .analysis-panel {
                    flex: 1;
                    padding: 20px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background: #fafafa;
                }
                video {
                    width: 100%;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Interview Recorder</h1>
                <div id="error-message"></div>
                <div class="video-container">
                    <video id="webcam" autoplay playsinline muted></video>
                    <div id="recordingStatus">Recording...</div>
                </div>
                <div class="timer" id="timer">00:00</div>
                <div class="controls">
                    <button id="startBtn" disabled>Start Recording</button>
                    <button id="stopBtn" disabled>Stop Recording</button>
                </div>
                <div class="results-container">
                    <div class="split-view">
                        <div class="video-player">
                            <h3>Processed Video</h3>
                            <video id="processedVideo" controls>
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        <div class="analysis-panel">
                            <h3>Analysis Results</h3>
                            <div id="results"></div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """
    Receives an uploaded video, saves it, and processes it with:
      1) VideoProcessor (image-based analysis, annotated video)
      2) AudioProcessor (audio extraction + transcription + audio emotion)
    """
    try:
        # Generate a unique ID for this video and save the file
        video_id = str(uuid.uuid4())
        video_path = UPLOAD_DIR / f"{video_id}.webm"

        async with aiofiles.open(video_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)

        # Mark the task as "processing"
        processing_tasks[video_id] = {
            "status": "processing",
            "start_time": datetime.now(),
            "path": str(video_path)
        }

        # 1) Process Video (face detection and image-based emotion analysis)
        video_results = video_processor.process_video(str(video_path))

        # 2) Extract and Analyze Audio
        audio_path = video_path.with_suffix(".wav")
        await audio_processor.extract_audio(str(video_path), str(audio_path))
        audio_results = audio_processor.analyze_audio(str(audio_path))

        # Merge both results
        combined_results = {
            "video_results": video_results,
            "audio_results": audio_results
        }

        # Update processing task status based on video processing outcome
        if video_results["status"] == "success":
            processing_tasks[video_id].update({
                "status": "completed",
                "results": combined_results,
                "completion_time": datetime.now()
            })
        else:
            processing_tasks[video_id].update({
                "status": "failed",
                "error": video_results.get("error", "Unknown error"),
                "completion_time": datetime.now()
            })

        return JSONResponse({
            "status": "success",
            "message": "Video uploaded and processed",
            "video_id": video_id
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-status/{video_id}")
async def get_video_status(video_id: str):
    """
    Returns the processing status of a given video.
    If completed, includes final analysis results.
    """
    if video_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Video not found")

    task = processing_tasks[video_id]
    if task["status"] == "completed":
        return {
            "status": "completed",
            "results": task["results"]
        }
    elif task["status"] == "failed":
        return {
            "status": "failed",
            "error": task.get("error", "Unknown error")
        }
    else:
        # Still processing
        return {
            "status": "processing",
            "started_at": task["start_time"].isoformat()
        }

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """
    Serves the annotated (processed) video if available.
    """
    if video_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Video not found")

    task = processing_tasks[video_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video processing not completed")

    video_data = task.get("results", {}).get("video_results", {})
    annotated_video_path = video_data.get("analysis", {}).get("annotated_video_path")

    if not annotated_video_path or not os.path.exists(annotated_video_path):
        raise HTTPException(status_code=404, detail="Annotated video file not found")

    return FileResponse(annotated_video_path)
