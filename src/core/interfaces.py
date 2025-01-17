from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class VideoProcessor(ABC):
    @abstractmethod
    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single video frame and return analysis results."""
        pass

class AudioProcessor(ABC):
    @abstractmethod
    async def process_chunk(self, audio_chunk: bytes) -> Dict[str, Any]:
        """Process an audio chunk and return analysis results."""
        pass

class InterviewManager(ABC):
    @abstractmethod
    async def generate_question(self, context: Dict[str, Any]) -> str:
        """Generate the next interview question based on context."""
        pass
    
    @abstractmethod
    async def analyze_response(self, 
                             response: str, 
                             emotions: Dict[str, float],
                             audio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's response including text, emotions, and audio metrics."""
        pass

class SessionManager(ABC):
    @abstractmethod
    async def create_session(self) -> str:
        """Create a new interview session and return session ID."""
        pass
    
    @abstractmethod
    async def save_session_data(self, 
                              session_id: str, 
                              data: Dict[str, Any]) -> None:
        """Save session data to storage."""
        pass
    
    @abstractmethod
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session summary and analysis."""
        pass