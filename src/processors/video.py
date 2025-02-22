import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

# Import MTCNN
MTCNN = None
try:
    # Try absolute import first
    from mtcnn.mtcnn import MTCNN
    logging.getLogger(__name__).info("Imported MTCNN successfully")
except ImportError:
    try:
        # Try pip package name
        from mtcnn import MTCNN
        logging.getLogger(__name__).info("Imported MTCNN successfully (alternative import)")
    except ImportError:
        MTCNN = None
        logging.getLogger(__name__).warning("MTCNN not found. Install with: pip install mtcnn")

# Conditionally import specific modules for your model
try:
    from ..models.emotions import EmotionRecognitionModel
    from ..models.arcface import ArcFaceBackbone
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import model modules - will attempt import when loading model")

# Default emotion mapping (can be overridden by model-specific mapping)
# Adjusted to 8 emotions based on your model's num_emotions=8
DEFAULT_EMOTION_MAPPING = {
    0: "angry",
    1: "disgust", 
    2: "fear", 
    3: "happy", 
    4: "sad", 
    5: "surprise", 
    6: "neutral",
    7: "contempt"  # Added contempt as the 8th emotion
}

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Enhanced Video Processor for face detection and emotion recognition.
    
    Features:
    - Supports MTCNN face detection (more accurate)
    - Loads custom emotion recognition models
    - Works with both CUDA and MPS (Apple Silicon)
    - Tracks faces across frames
    - Provides detailed emotion analytics
    """
    def __init__(self, model_path: str = None, device: str = None):
        # Determine device for inference
        self.device = self._get_device(device)
        device_label = "GPU" if self.device in ["cuda", "mps"] else "CPU"
        logger.info(f"Using device for video processing: {device_label} ({self.device})")
        
        # Initialize the face detection system
        self._init_face_detector()
        
        # Initialize the emotion recognition model
        self.model = None
        self.transform = None
        self.emotion_mapping = DEFAULT_EMOTION_MAPPING
        if model_path:
            self._load_emotion_model(model_path)
        
        # Prepare output directory for processed videos and results
        self.output_dir = Path("processed_videos")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Tracking variables for multiple faces
        self.tracked_faces = OrderedDict()
        self.next_face_id = 0
        self.face_emotions = {}
        
        logger.info("VideoProcessor initialized successfully")

    def _get_device(self, device: Optional[str]) -> str:
        """
        Determine the appropriate computing device.
        
        Args:
            device: Optional device specification (cuda, mps, or cpu)
            
        Returns:
            String representing the device to use
        """
        if device is not None:
            return device
            
        # Auto-detect available devices with detailed logging
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS available (Apple Silicon)")
            return "mps"
        else:
            logger.info("Using CPU for processing")
            return "cpu"

    def _init_face_detector(self):
        """
        Initialize the face detection system using MTCNN or fall back to Haar cascade.
        """
        try:
            # First try MTCNN (better quality but requires additional package)
            if MTCNN is not None:
                self.use_mtcnn = True
                self.face_detector = MTCNN()
                logger.info("MTCNN face detector initialized successfully")
            else:
                # Fall back to OpenCV's Haar cascade
                self.use_mtcnn = False
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                if self.face_cascade.empty():
                    logger.error("Failed to load Haar cascade for face detection")
                    raise RuntimeError("Failed to load face detection model")
                    
                logger.info("Face detector initialized successfully (using Haar cascade as fallback)")
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
            raise

    def _load_emotion_model(self, model_path: str):
        """
        Load a pre-trained emotion recognition model.
        
        Args:
            model_path: Path to the PyTorch model file
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading emotion model from: {model_path}")
            
            # Import the model class from your models module
            # This approach assumes the model architecture is accessible via import
            try:
                try:
                    from ..models.emotions import EmotionRecognitionModel
                except ImportError:
                    # Try absolute import
                    from src.models.emotions import EmotionRecognitionModel
                logger.info("Successfully imported EmotionRecognitionModel")
            except ImportError as e:
                logger.error(f"Failed to import EmotionRecognitionModel: {e}")
                raise ImportError(f"Could not import EmotionRecognitionModel: {e}")
            
            # Create model instance with your complex architecture
            # Using the parameters based on your implementation
            # Note: Setting freeze_backbone=False as in your original code
            self.model = EmotionRecognitionModel(embedding_size=512, num_emotions=8, freeze_backbone=False)
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                
                # Try to load emotion mapping if present in the checkpoint
                if "emotion_mapping" in checkpoint:
                    self.emotion_mapping = checkpoint["emotion_mapping"]
                    logger.info(f"Loaded emotion mapping from checkpoint: {self.emotion_mapping}")
            else:
                state_dict = checkpoint
            
            # Load the state dict with appropriate error handling
            try:
                self.model.load_state_dict(state_dict)
                logger.info("Successfully loaded model weights")
            except Exception as e:
                logger.warning(f"Error loading exact state dict: {e}")
                logger.warning("Attempting to load with strict=False")
                # Try loading with strict=False to handle partial model loading
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                logger.warning(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                
            # Move model to appropriate device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Standard image transformations for deep learning models
            # Using ImageNet normalization as per common practice with ArcFace models
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),  # ArcFace typically uses 112x112
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]    # ImageNet stds
                )
            ])
            
            logger.info("Emotion recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            self.model = None
            raise

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file for face detection and emotion analysis.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            # Verify that the input video exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Open the video file
            logger.info(f"Processing video: {video_path}")
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
            video_id = Path(video_path).stem
            annotated_path = str(self.output_dir / f"{video_id}_annotated.mp4")
            out = self._get_video_writer(annotated_path, fps, (frame_width, frame_height))

            # Variables for emotion analysis and face tracking
            emotions_over_time: List[Dict[str, Any]] = []
            frame_interval = 1  # Process emotions on every frame for better accuracy
            self.tracked_faces = OrderedDict()
            self.next_face_id = 0
            self.face_emotions = {}

            # Initialize a progress counter
            processed_frames = 0
            
            # Process the video frame by frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get current frame index
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                processed_frames += 1
                
                # Detect faces
                detected_faces = self._detect_faces(frame)
                
                # Update face tracking
                self._update_face_tracking(detected_faces)
                
                # Process emotions on every frame if model is loaded
                if self.model is not None:
                    self._process_emotions(frame, frame_idx, fps, emotions_over_time)
                
                # Annotate the frame with bounding boxes and emotion labels
                self._annotate_frame(frame)
                
                # Write the processed frame to the output video
                out.write(frame)
                
                # Display progress periodically
                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%)")

            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Aggregate the results
            visual_results = self._aggregate_results(emotions_over_time, duration)
            
            # Save the results to disk
            result_dict = {
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
            
            # Save results to file
            self.save_results(video_id, result_dict)
            
            # Log the path to the annotated video
            logger.info(f"Annotated video saved at: {annotated_path}")
            logger.info(f"Output directory is: {self.output_dir}")
            
            logger.info(f"Video processing completed: {video_path}")
            return result_dict

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using either MTCNN or Haar cascade.
        
        Args:
            frame: Frame image as numpy array
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        try:
            if self.use_mtcnn:
                # Convert to RGB for MTCNN (which expects RGB images)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces using MTCNN
                detections = self.face_detector.detect_faces(rgb_frame)
                
                # Extract bounding boxes
                faces = []
                for detection in detections:
                    x, y, w, h = detection["box"]
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    # Add to faces list
                    if w > 0 and h > 0:
                        faces.append((x, y, w, h))
                return faces
            else:
                # Use Haar cascade (fallback method)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.05,  # Lower value for better detection (1.05 instead of 1.1)
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                
                # Convert to list of tuples
                if len(faces) > 0:
                    return [(x, y, w, h) for (x, y, w, h) in faces]
                else:
                    return []
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []

    def _update_face_tracking(self, detected_faces: List[Tuple[int, int, int, int]]):
        """
        Update the face tracking across frames.
        
        Args:
            detected_faces: List of (x, y, w, h) tuples for detected faces
        """
        updated_tracked_faces = OrderedDict()
        
        for (x, y, w, h) in detected_faces:
            matched_id = None
            min_dist = float('inf')
            
            # Try to match with existing faces by position
            for face_id, (tx, ty, tw, th) in self.tracked_faces.items():
                # Calculate center points
                curr_center_x = x + w/2
                curr_center_y = y + h/2
                tracked_center_x = tx + tw/2
                tracked_center_y = ty + th/2
                
                # Euclidean distance between centers
                dist = np.hypot(curr_center_x - tracked_center_x, curr_center_y - tracked_center_y)
                
                # Match if distance is below threshold (50 pixels)
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    matched_id = face_id
            
            if matched_id is not None:
                # Update position of existing face
                updated_tracked_faces[matched_id] = (x, y, w, h)
            else:
                # Add new face
                updated_tracked_faces[self.next_face_id] = (x, y, w, h)
                self.next_face_id += 1
        
        # Update tracked faces
        self.tracked_faces = updated_tracked_faces

    def _process_emotions(self, frame: np.ndarray, frame_idx: int, fps: float, emotions_over_time: List[Dict[str, Any]]):
        """
        Process emotions for all tracked faces in the current frame.
        
        Args:
            frame: Current video frame
            frame_idx: Current frame index
            fps: Frames per second
            emotions_over_time: List to store emotion data over time
        """
        timestamp_sec = frame_idx / fps
        
        for face_id, (x, y, w, h) in self.tracked_faces.items():
            # Ensure coordinates are within image bounds
            x, y, w, h = max(0, x), max(0, y), w, h
            
            # Apply minimum size constraints (skip if face is too small)
            if w < 20 or h < 20:
                continue
                
            # Ensure we don't exceed image boundaries
            h_max = min(y + h, frame.shape[0])
            w_max = min(x + w, frame.shape[1])
            
            # Skip if adjusted size is too small
            if h_max - y < 10 or w_max - x < 10:
                continue
                
            # Extract face region
            face_region = frame[y:h_max, x:w_max]
            
            # Skip empty regions
            if face_region.size == 0:
                continue
            
            # Convert to RGB for the model (important for consistent results)
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(rgb_face)
            
            # Apply transformations and run the model
            face_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    # Get class probabilities
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Get predicted class
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    predicted_emotion = self.emotion_mapping.get(predicted_class, f"emotion_{predicted_class}")
                    
                    # Get emotion predictions and scores for all classes
                    emotions = {}
                    for i, prob in enumerate(probabilities):
                        emotion_label = self.emotion_mapping.get(i, f"emotion_{i}")
                        emotions[emotion_label] = float(prob)
                    
                    # Store emotions for this face
                    self.face_emotions[face_id] = {
                        "emotions": emotions,
                        "top_emotion": predicted_emotion,
                        "top_score": float(probabilities[predicted_class])
                    }
                    
                    # Add to timeline
                    emotions_over_time.append({
                        "timestamp": float(timestamp_sec),
                        "emotions": emotions,
                        "face_position": (int(x), int(y), int(w), int(h)),
                        "face_id": face_id,
                        "top_emotion": predicted_emotion
                    })
            except Exception as e:
                logger.error(f"Error in emotion prediction: {str(e)}")

    def _annotate_frame(self, frame: np.ndarray):
        """
        Annotate the frame with bounding boxes and emotion labels.
        
        Args:
            frame: Current video frame to annotate
        """
        for face_id, (x, y, w, h) in self.tracked_faces.items():
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get emotions for this face
            face_data = self.face_emotions.get(face_id)
            if face_data:
                # Add emotion label with score
                label = f"{face_data['top_emotion']} ({face_data['top_score']:.2f})"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # If no emotion data yet, just show face ID
                cv2.putText(frame, f"Face {face_id}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _aggregate_results(self, emotions_over_time: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
        """
        Aggregate emotion data across all frames.
        
        Args:
            emotions_over_time: List of emotion data points
            duration: Video duration in seconds
            
        Returns:
            Dictionary with aggregated emotion metrics
        """
        if not emotions_over_time:
            return {
                "average_emotions": {},
                "peak_emotions": {},
                "timeline": []
            }

        # Group emotions by face ID
        faces_data = {}
        for entry in emotions_over_time:
            face_id = entry["face_id"]
            if face_id not in faces_data:
                faces_data[face_id] = []
            faces_data[face_id].append(entry)
        
        all_average_emotions = {}
        all_peak_emotions = {}
        timeline = []
        
        # Process each face separately
        for face_id, face_entries in faces_data.items():
            # Calculate average emotions for this face
            emotion_sums = {}
            emotion_counts = {}
            
            for entry in face_entries:
                for emotion, score in entry["emotions"].items():
                    if emotion not in emotion_sums:
                        emotion_sums[emotion] = 0
                        emotion_counts[emotion] = 0
                    emotion_sums[emotion] += score
                    emotion_counts[emotion] += 1
            
            # Calculate averages
            face_average_emotions = {
                emotion: emotion_sums[emotion] / emotion_counts[emotion]
                for emotion in emotion_sums
            }
            
            # Find peak emotions
            face_peak_emotions = {}
            for entry in face_entries:
                for emotion, score in entry["emotions"].items():
                    if emotion not in face_peak_emotions or score > face_peak_emotions[emotion]["score"]:
                        face_peak_emotions[emotion] = {
                            "score": score,
                            "timestamp": entry["timestamp"]
                        }
            
            # Add to global aggregates
            all_average_emotions[f"face_{face_id}"] = face_average_emotions
            all_peak_emotions[f"face_{face_id}"] = face_peak_emotions
            
            # Add all entries to timeline
            timeline.extend(face_entries)
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return {
            "average_emotions": all_average_emotions,
            "peak_emotions": all_peak_emotions,
            "timeline": timeline,
            "face_count": len(faces_data),
            "dominant_emotions": self._get_dominant_emotions(all_average_emotions)
        }
    
    def _get_dominant_emotions(self, all_average_emotions: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Determine the dominant emotion for each face.
        
        Args:
            all_average_emotions: Dictionary of average emotions by face
            
        Returns:
            Dictionary mapping face IDs to dominant emotions
        """
        dominant_emotions = {}
        for face_id, emotions in all_average_emotions.items():
            if emotions:
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                dominant_emotions[face_id] = {
                    "emotion": dominant_emotion[0],
                    "score": dominant_emotion[1]
                }
        return dominant_emotions

    def _get_video_writer(self, output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """
        Create a video writer with appropriate codec settings.
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second
            frame_size: (width, height) of the video frames
            
        Returns:
            Initialized VideoWriter object
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get codec based on file extension
            ext = os.path.splitext(output_path)[1].lower()
            
            if ext == '.mp4':
                # Default to MP4V for MP4 format (broadly compatible)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif ext == '.avi':
                # XVID for AVI format
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                # Default to MP4V for unknown extensions
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Create the video writer
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
        Save analysis results to a JSON file.
        
        Args:
            video_id: Identifier for the video
            results: Dictionary of analysis results
            
        Returns:
            Path to the saved results file
        """
        try:
            self.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Main results file
            results_path = self.output_dir / f"{video_id}_results.json"
            with results_path.open('w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Saved video results for {video_id}")
            return str(results_path)
        except Exception as e:
            logger.error(f"Failed to save video results: {str(e)}")
            raise

    def get_results(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve saved analysis results.
        
        Args:
            video_id: Identifier for the video
            
        Returns:
            Dictionary of analysis results or None if not found
        """
        results_path = self.output_dir / f"{video_id}_results.json"
        if results_path.exists():
            with results_path.open('r') as f:
                return json.load(f)
        return None
        
    def process_webcam(self, output_file: str = None, display: bool = True, max_duration: int = None) -> str:
        """
        Process video directly from webcam with real-time emotion analysis.
        
        Args:
            output_file: Optional path to save the output video
            display: Whether to display the video during processing
            max_duration: Maximum recording duration in seconds (None for unlimited)
            
        Returns:
            Path to the saved output video (if output_file was specified)
        """
        try:
            # Initialize webcam capture
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("Could not open webcam")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Default to 30 FPS if not properly reported
                
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer if output file is specified
            out = None
            if output_file:
                out = self._get_video_writer(output_file, fps, (frame_width, frame_height))
            
            # Reset tracking state
            self.tracked_faces = OrderedDict()
            self.next_face_id = 0
            self.face_emotions = {}
            
            # Variables for emotion analysis
            emotions_over_time = []
            frame_count = 0
            start_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            logger.info("Starting webcam processing. Press 'q' to stop.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update frame count and calculate elapsed time
                frame_count += 1
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed_time = current_time - start_time
                
                # Check if we've reached the maximum duration
                if max_duration and elapsed_time > max_duration:
                    logger.info(f"Reached maximum duration of {max_duration} seconds")
                    break
                
                # Detect faces
                detected_faces = self._detect_faces(frame)
                
                # Update face tracking
                self._update_face_tracking(detected_faces)
                
                # Process emotions (on every frame for better performance)
                if self.model is not None:
                    self._process_emotions(frame, frame_count, fps, emotions_over_time)
                
                # Annotate the frame
                self._annotate_frame(frame)
                
                # Add time counter to the frame
                cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Write frame to output if specified
                if out:
                    out.write(frame)
                
                # Display the frame if requested
                if display:
                    cv2.imshow('Emotion Recognition', frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # If output file was specified, save the analysis results
            if output_file:
                video_id = Path(output_file).stem
                
                # Calculate duration based on frame count and FPS
                duration = frame_count / fps
                
                # Aggregate results
                visual_results = self._aggregate_results(emotions_over_time, duration)
                
                # Save results to file
                results_dict = {
                    "status": "success",
                    "video_info": {
                        "duration": duration,
                        "total_frames": frame_count,
                        "fps": fps,
                        "width": frame_width,
                        "height": frame_height
                    },
                    "analysis": {
                        "visual_analysis": visual_results,
                        "annotated_video_path": output_file
                    }
                }
                
                self.save_results(video_id, results_dict)
                logger.info(f"Webcam processing completed and saved to {output_file}")
                
                return output_file
            else:
                logger.info("Webcam processing completed (no output file specified)")
                return None
                
        except Exception as e:
            logger.error(f"Error processing webcam: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Ask user to select device
    print("Select computing device:")
    print("1. Auto-detect (recommended)")
    print("2. CUDA (NVIDIA GPU)")
    print("3. MPS (Apple Silicon)")
    print("4. CPU")
    
    choice = input("Enter your choice (1-4): ")
    
    device = None
    if choice == '2':
        device = "cuda"
    elif choice == '3':
        device = "mps"
    elif choice == '4':
        device = "cpu"
    # For choice 1 or invalid input, device remains None for auto-detection
    
    model_path = "best_emotion_model.pth"
    processor = VideoProcessor(model_path=model_path, device=device)
    
    # Ask what to process
    print("\nWhat would you like to do?")
    print("1. Process video file")
    print("2. Process from webcam")
    
    action = input("Enter your choice (1-2): ")
    
    if action == '1':
        video_path = input("Enter video path: ")
        result = processor.process_video(video_path)
        print(f"Processing complete. Results saved to: {result.get('analysis', {}).get('annotated_video_path', 'unknown')}")
    else:
        output_file = input("Enter output file name (or press Enter for default 'webcam_output.mp4'): ")
        if not output_file:
            output_file = "webcam_output.mp4"
            
        max_duration = input("Enter maximum recording duration in seconds (or press Enter for 60s): ")
        if not max_duration:
            max_duration = 60
        else:
            max_duration = int(max_duration)
            
        processor.process_webcam(output_file=output_file, max_duration=max_duration)
        print(f"Recording complete. Output saved to: {output_file}")