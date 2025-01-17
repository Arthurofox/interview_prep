from fastapi import APIRouter, Depends
from src.core.config import Settings, get_settings

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "app_name": settings.APP_NAME
    }