# Interview Preparation Tool 🎯

Real-time interview analysis tool that provides emotion detection and speech-to-text capabilities to help users improve their interview skills.

## Features 🌟

- Real-time facial emotion detection
- Live speech-to-text transcription
- WebSocket-based real-time communication
- Clean architecture design

## Tech Stack 🛠️

- **Backend**:
  - FastAPI
  - Python 3.11+
  - MediaPipe for face detection
  - Hugging Face Transformers for emotion analysis
  - Faster Whisper for speech-to-text

- **Frontend**:
  - Pure JavaScript
  - HTML5 WebSocket API
  - Canvas API for video processing

## Getting Started 🚀

### Prerequisites

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

### Installation

```bash
# Install required packages
pip install fastapi uvicorn python-multipart pillow deepface tensorflow mediapipe opencv-python-headless faster-whisper torch transformers
```

### Running the Application

```bash
python run.py
```

Visit `http://localhost:8000` in your web browser.

## Project Structure 📁

```
interview_prep/
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── interfaces.py
│   │   └── exceptions.py
│   ├── interface/
│   │   └── api/
│   │       ├── main.py
│   │       └── routers/
│   └── processors/
│       ├── video.py
│       └── audio.py
└── run.py
```

## Development Status 🔄

Current implementation provides basic functionality with:
- Face detection and emotion analysis
- Audio transcription
- Real-time feedback

Future improvements planned:
- M2 chip optimization using Metal API
- CoreML integration for better performance
- Enhanced emotion detection accuracy
- Improved real-time processing

## Contributing 🤝

Feel free to open issues and pull requests for improvements!