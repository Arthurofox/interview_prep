import asyncio
import logging
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import librosa
import soundfile as sf
import subprocess
from transformers import pipeline, logging as transformers_logging
from faster_whisper import WhisperModel

# Suppress various library warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*weights of the model.*')
warnings.filterwarnings('ignore', message='.*gradient_checkpointing.*')

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio extraction from a video, full transcription via Whisper,
    and audio-based emotion analysis (no partial chunking).
    """
    def __init__(self, device: Optional[str] = None):
        # Determine device for inference
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        device_label = "GPU" if device == "cuda" else "CPU"
        logger.info(f"Initializing AudioProcessor on {device_label}")

        # Set compute precision for Whisper based on device
        compute_type = "float16" if self.device == "cuda" else "int8"

        # Initialize Whisper model
        try:
            self.whisper_model = WhisperModel(
                model_size_or_path="tiny",
                device=self.device,
                compute_type=compute_type
            )
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {e}")
            self.whisper_model = None

        # Initialize audio emotion recognition pipeline
        try:
            self.audio_emotion_model = pipeline(
                task="audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if self.device == "cuda" else -1,
                truncation=True
            )
            logger.info("Audio emotion recognition model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio emotion model: {e}")
            self.audio_emotion_model = None

    async def extract_audio(self, video_path: str, audio_path: str) -> None:
        """
        Extracts raw audio from the given video file using FFmpeg, saving it to 'audio_path'.
        This asynchronous method spawns a subprocess to perform the extraction.
        """
        logger.info(f"Extracting audio from '{video_path}' to '{audio_path}'")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",                      # No video
            "-acodec", "pcm_s16le",     # PCM 16-bit little-endian
            "-ar", "16000",             # 16 kHz sample rate
            "-ac", "1",                 # Mono channel
            "-y",                       # Overwrite output file if exists
            audio_path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="ignore")
                logger.error(f"FFmpeg error: {error_msg}")
                raise RuntimeError("Failed to extract audio from video")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyzes the audio file for transcription and emotion detection.
        Returns a dictionary with transcription text and emotion analysis results.
        """
        results: Dict[str, Any] = {
            "transcription": "",
            "emotions": {
                "average_emotions": {},
                "peak_emotions": {},
                "timeline": []
            }
        }

        if not self.whisper_model:
            logger.error("Whisper model is not initialized.")
            results["error"] = "Whisper model not available"
            return results

        logger.info(f"Analyzing audio file: {audio_path}")

        try:
            # Load entire audio file at 16 kHz
            audio_data, sr = librosa.load(audio_path, sr=16000)
            if audio_data.size == 0:
                logger.warning("Loaded audio is empty.")
                return results

            # Transcription using Whisper: set language to French ("fr")
            segments, info = self.whisper_model.transcribe(audio_data, language="fr", beam_size=5)
            full_text = " ".join(seg.text for seg in segments).strip()
            results["transcription"] = full_text
            logger.info(f"Transcription: {full_text}")

            if not self.audio_emotion_model:
                logger.warning("Audio emotion model is not initialized.")
                return results

            # Run emotion classification on the entire waveform
            logger.info("Running audio emotion recognition on entire waveform.")
            emotions = self.audio_emotion_model(
                {"array": audio_data.astype(np.float32), "sampling_rate": 16000},
                top_k=5
            )
            logger.info(f"Emotion Results: {emotions}")

            if emotions:
                emotion_dict = {item["label"]: float(item["score"]) for item in emotions}
                results["emotions"]["timeline"].append({
                    "timestamp": 0.0,
                    "emotions": emotion_dict
                })
                results["emotions"]["average_emotions"] = emotion_dict
                results["emotions"]["peak_emotions"] = {
                    label: {"score": score, "timestamp": 0.0} for label, score in emotion_dict.items()
                }

            # Save results to file
            audio_id = Path(audio_path).stem
            output_json = self.save_results(audio_id, results)
            logger.info(f"Results saved to '{output_json}'")

        except Exception as e:
            logger.error(f"Error during audio analysis: {e}")
            results["error"] = str(e)

        return results

    def save_results(self, audio_id: str, results: Dict[str, Any], output_dir: str = "processed_audio") -> str:
        """
        Saves the analysis results to a JSON file in the specified output directory.
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            results_file = output_path / f"{audio_id}_audio_results.json"

            with results_file.open('w') as f:
                json.dump(results, f, indent=4)

            logger.info(f"Saved audio results for '{audio_id}' to '{results_file}'")
            return str(results_file)
        except Exception as e:
            logger.error(f"Error saving audio results: {e}")
            raise

    def reset(self):
        """
        No-op placeholder if you plan to store streaming or chunk-based state.
        """
        pass
