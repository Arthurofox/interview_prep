from typing import Dict, Any
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import logging
from transformers import pipeline
import torch
from PIL import Image
from src.core.config import get_settings

logger = logging.getLogger(__name__)

class MediaPipeVideoProcessor:
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh  # Use face mesh for better accuracy
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Initialize emotion detection pipeline
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        try:
            self.emotion_detector = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                device=device
            )
            logger.info("Emotion detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing emotion detector: {str(e)}")
            self.emotion_detector = None
        
        # Frame counter and processing settings
        self.frame_counter = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for emotions
        
        logger.info("MediaPipeVideoProcessor initialized successfully")
    

    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a video frame and return face and emotion analysis with visual cues."""
        self.frame_counter += 1

        result = {
            "frame_number": self.frame_counter,
            "faces_detected": 0,
            "emotions": None
        }

        try:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and landmarks using MediaPipe face mesh
            mesh_results = self.face_mesh.process(rgb_frame)

            if mesh_results.multi_face_landmarks:
                result["faces_detected"] = len(mesh_results.multi_face_landmarks)

                # Process emotions every N frames
                if self.frame_counter % self.process_every_n_frames == 0:
                    # Iterate over detected faces
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        # Draw facial landmarks and bounding box
                        h, w = frame.shape[:2]

                        # Bounding box coordinates
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0
                        for lm in face_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min = min(x_min, x)
                            y_min = min(y_min, y)
                            x_max = max(x_max, x)
                            y_max = max(y_max, y)

                        # Add padding
                        padding = 20
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(w, x_max + padding)
                        y_max = min(h, y_max + padding)

                        # Draw the bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Draw the facial landmarks
                        for lm in face_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

                        # Crop and process face image for emotion detection
                        face_image = frame[y_min:y_max, x_min:x_max]
                        if face_image.size > 0:
                            emotions = await self._analyze_emotions(face_image)
                            result["emotions"] = emotions

                        # Annotate detected emotions
                        if result["emotions"]:
                            emotion_text = ", ".join(
                                f"{emotion}: {score:.2f}" for emotion, score in result["emotions"].items()
                            )
                            cv2.putText(
                                frame,
                                emotion_text,
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            result["error"] = str(e)

        return result

    
    async def _analyze_emotions(self, face_image: np.ndarray) -> Dict[str, float]:
        """Analyze emotions in a face image using the Hugging Face pipeline."""
        try:
            # Convert numpy array to PIL Image
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_image_rgb)
            
            # Run emotion detection in executor to avoid blocking
            emotions_output = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.emotion_detector,
                pil_image
            )
            
            # Convert the output to a dictionary of emotion probabilities
            emotions_dict = {
                item['label']: float(item['score'])
                for item in emotions_output
            }
            
            return emotions_dict
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return {}
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
