from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

@dataclass
class Question:
    text: str
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class Response:
    text: str
    timestamp: datetime
    emotions: Dict[str, float]
    analysis: Dict[str, Any]

@dataclass
class InterviewSession:
    id: str
    start_time: datetime
    end_time: datetime | None = None
    questions: List[Question] = field(default_factory=list)
    responses: List[Response] = field(default_factory=list)