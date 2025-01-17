import numpy as np
from typing import Dict, Any
import torch
from faster_whisper import WhisperModel
import logging
from src.core.config import get_settings

logger = logging.getLogger(__name__)

class SpeechAnalyzer:
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Initializing Whisper model on {device} with compute type {compute_type}")
        
        # Initialize Whisper with the specified model size
        model_size = "tiny"  # Can be "tiny", "base", "small", "medium", "large"
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        
        # Initialize buffer for continuous audio
        self.audio_buffer = np.array([], dtype=np.float32)
        
        logger.info("SpeechAnalyzer initialized successfully")
    
    async def analyze_chunk(self, audio_chunk: bytes) -> Dict[str, Any]:
        """Analyze an audio chunk for speech-to-text."""
        try:
            # Ensure the audio chunk size is correct
            chunk_size = len(audio_chunk)
            if chunk_size % 4 != 0:  # 4 bytes per float32
                padding_size = 4 - (chunk_size % 4)
                audio_chunk = audio_chunk + b'\x00' * padding_size
            
            # Convert bytes to numpy array
            chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer = np.append(self.audio_buffer, chunk_np)
            
            result = {
                "transcription": None,
                "buffer_size": len(self.audio_buffer),
                "error": None
            }
            
            # Only process if buffer is large enough
            if len(self.audio_buffer) >= self.settings.AUDIO_CHUNK_SIZE:
                try:
                    # Transcribe
                    segments, _ = self.model.transcribe(
                        self.audio_buffer,
                        language="en",
                        beam_size=5
                    )
                    
                    transcription = " ".join(segment.text for segment in segments)
                    if transcription.strip():
                        result["transcription"] = transcription.strip()
                    
                    # Clear buffer after successful transcription
                    self.audio_buffer = np.array([], dtype=np.float32)
                    
                except Exception as e:
                    logger.error(f"Transcription error: {str(e)}")
                    result["error"] = f"Speech-to-text failed: {str(e)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            return {
                "transcription": None,
                "buffer_size": 0,
                "error": f"Audio analysis failed: {str(e)}"
            }