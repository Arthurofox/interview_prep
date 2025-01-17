from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from core.config import get_settings
from core.logging import setup_logging

def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings)
    
    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        version="0.1.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with actual origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from .routers import interview, health
    app.include_router(health.router, prefix=settings.API_PREFIX)
    app.include_router(interview.router, prefix=settings.API_PREFIX)
    
    return app