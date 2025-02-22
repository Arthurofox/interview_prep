from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import logging
from transformers import pipeline
import torch
from PIL import Image
from pathlib import Path
import json
import platform
from collections import OrderedDict
from src.config import get_settings

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video processing for face detection and image-based emotion analysis.
    Does not perform audio extraction or audio-based analysis.
    """
    def __init__(self):
        # Determine device for inference (using GPU if available)
        self.device = 0 if torch.cuda.is_available() else -1
        device_label = "GPU" if self.device == 0 else "CPU"
        logger.info(f"Using device: {device_label}")

        # Initialize the image-based emotion detection pipeline
        # (Using the original model identifier provided, which is an empty string.
        #  Replace "" with a valid model identifier later if needed.)
        try:
            self.emotion_detector = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                device=self.device
            )
            logger.info("Image emotion detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing emotion detector: {str(e)}")
            self.emotion_detector = None

        # Initialize face detection (Haar cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("Failed to load Haar cascade classifier for face detection")

        # Load application settings (if any)
        self.settings = get_settings()

        # Prepare output directory for processed videos and results
        self.output_dir = Path("processed_videos")
        self.output_dir.mkdir(exist_ok=True)

        # Log system information
        self.system = platform.system()
        logger.info(f"Running on {self.system}")

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyzes the video to detect faces and classify their emotions.
        Uses a tracking system to maintain emotion information for each face.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Read video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0:
                logger.warning("FPS reported as 0 or negative; using fallback = 30.0")
                fps = 30.0
            duration = float(total_frames / fps) if total_frames > 0 else 0.0

            # Create annotated output video path and writer
            annotated_path = str(
                Path(video_path)
                .with_stem(Path(video_path).stem + "_annotated")
                .with_suffix(".mp4")
            )
            out = self._get_video_writer(annotated_path, fps, (frame_width, frame_height))

            # Variables for emotion analysis and face tracking
            emotions_over_time: List[Dict[str, Any]] = []
            frame_interval = int(fps * 1)  # Process emotions every second
            tracked_faces: OrderedDict[int, Tuple[int, int, int, int]] = OrderedDict()
            next_face_id = 0
            face_emotions: Dict[int, Dict[str, float]] = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get current frame index
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Convert frame to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
                detected_faces = [(x, y, w, h) for (x, y, w, h) in detected]

                # Match detected faces to previously tracked faces
                updated_tracked_faces: OrderedDict[int, Tuple[int, int, int, int]] = OrderedDict()
                for (x, y, w, h) in detected_faces:
                    matched_id = None
                    min_dist = float('inf')
                    for face_id, (tx, ty, tw, th) in tracked_faces.items():
                        # Calculate Euclidean distance between top-left points
                        dist = np.hypot(tx - x, ty - y)
                        if dist < min_dist and dist < 50:
                            min_dist = dist
                            matched_id = face_id
                    if matched_id is not None:
                        updated_tracked_faces[matched_id] = (x, y, w, h)
                    else:
                        updated_tracked_faces[next_face_id] = (x, y, w, h)
                        next_face_id += 1

                tracked_faces = updated_tracked_faces

                # Process emotions at defined intervals (once per second)
                if frame_idx % frame_interval == 0:
                    for face_id, (x, y, w, h) in tracked_faces.items():
                        face_region = frame[y:y+h, x:x+w]
                        if face_region.size == 0:
                            continue

                        # Convert face region from BGR to RGB and wrap with PIL Image
                        pil_face = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                        emotions = self._analyze_emotions(pil_face)
                        if emotions:
                            face_emotions[face_id] = emotions
                            timestamp_sec = frame_idx / fps
                            emotions_over_time.append({
                                "timestamp": float(timestamp_sec),
                                "emotions": emotions,
                                "face_position": (int(x), int(y), int(w), int(h)),
                                "face_id": face_id
                            })

                # Draw annotations for every frame
                for face_id, (x, y, w, h) in tracked_faces.items():
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Retrieve last known emotions for this face
                    emotions = face_emotions.get(face_id, None)
                    if emotions:
                        # Compute the emotion with the highest score
                        top_emotion = max(emotions.items(), key=lambda item: item[1])
                        label = f"{top_emotion[0]} ({top_emotion[1]:.0%})"
                        logger.debug(f"Face {face_id}: {label}")
                    else:
                        # Fallback if no emotion is detected
                        label = "No emotion"
                        logger.debug(f"Face {face_id}: No emotion detected")
                    
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Write frame to output video
                out.write(frame)

            cap.release()
            out.release()

            visual_results = self._aggregate_results(emotions_over_time, duration)
            return {
                "status": "success",
                "video_info": {
                    "duration": duration,
                    "total_frames": total_frames,
                    "fps": fps,
                    "width": frame_width,
                    "height": frame_height
                },
                "analysis": {
                    "visual_analysis": visual_results,
                    "annotated_video_path": annotated_path
                }
            }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _analyze_emotions(self, face_image: Image) -> Optional[Dict[str, float]]:
        """
        Runs the Hugging Face image-classification pipeline
        to detect emotions in a face image.
        """
        if self.emotion_detector is None:
            return None

        try:
            results = self.emotion_detector(face_image)
            logger.info(f"Emotion detector raw results: {results}")
            return {item["label"]: float(item["score"]) for item in results}
        except Exception as e:
            logger.error(f"Error in emotion classification: {str(e)}")
            return None

    def _aggregate_results(self, emotions_over_time: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
        """
        Aggregates emotion data over all sampled frames.
        Returns average emotions, peak emotions, and a timeline.
        """
        if not emotions_over_time:
            return {
                "average_emotions": {},
                "peak_emotions": {},
                "timeline": []
            }

        emotion_sums: Dict[str, float] = {}
        emotion_counts: Dict[str, int] = {}
        timeline: List[Dict[str, Any]] = []

        for entry in emotions_over_time:
            timestamp = float(entry["timestamp"])
            face_emotions = {k: float(v) for k, v in entry["emotions"].items()}

            timeline.append({
                "timestamp": timestamp,
                "emotions": face_emotions,
                "face_position": entry["face_position"]
            })

            for label, score in face_emotions.items():
                emotion_sums[label] = emotion_sums.get(label, 0.0) + score
                emotion_counts[label] = emotion_counts.get(label, 0) + 1

        average_emotions = {
            label: emotion_sums[label] / emotion_counts[label]
            for label in emotion_sums
        }

        peak_emotions: Dict[str, Dict[str, float]] = {}
        for data in timeline:
            for label, score in data["emotions"].items():
                if label not in peak_emotions or score > peak_emotions[label]["score"]:
                    peak_emotions[label] = {"score": score, "timestamp": data["timestamp"]}

        return {
            "average_emotions": average_emotions,
            "peak_emotions": peak_emotions,
            "timeline": timeline
        }

    def _get_video_writer(self, output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """
        Creates and returns a VideoWriter with proper MP4 codec configuration.
        """
        try:
            # Use MP4V codec for broad compatibility (including on macOS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (int(frame_size[0]), int(frame_size[1]))
            writer = cv2.VideoWriter(
                filename=output_path,
                fourcc=fourcc,
                fps=float(fps),
                frameSize=frame_size,
                isColor=True
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")
            return writer
        except Exception as e:
            logger.error(f"Video writer error: {str(e)}")
            raise

    def save_results(self, video_id: str, results: Dict[str, Any]) -> str:
        """
        Saves result dictionaries to JSON files in the output directory.
        """
        try:
            self.output_dir.mkdir(exist_ok=True, parents=True)

            # Main results file (all data)
            results_path = self.output_dir / f"{video_id}_results.json"
            with results_path.open('w') as f:
                json.dump(results, f, indent=4)

            # Optional script file (transcription) — empty by default here
            script_path = self.output_dir / f"{video_id}_script.json"
            with script_path.open('w') as f:
                json.dump({"transcription": ""}, f, indent=4)

            # Optional audio analysis file — empty by default here
            audio_analysis_path = self.output_dir / f"{video_id}_audio_analysis.json"
            with audio_analysis_path.open("w") as f:
                json.dump({}, f, indent=4)

            logger.info(f"Saved video-only results for {video_id}")
            return str(results_path)
        except Exception as e:
            logger.error(f"Failed to save video results: {str(e)}")
            raise

    def get_results(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves saved results (if any) from the output directory.
        """
        results_path = self.output_dir / f"{video_id}_results.json"
        if results_path.exists():
            with results_path.open('r') as f:
                return json.load(f)
        return None
