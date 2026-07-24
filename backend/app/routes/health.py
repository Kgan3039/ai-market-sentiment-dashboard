"""
Health Check Route

Simple endpoint to verify the API is running and operational.
"""

from fastapi import APIRouter
from app.config import settings
from app.phase0.schemas import HealthResponse

router = APIRouter()


@router.get("/test", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse: API status and version information

    Example:
        GET /test
        Response: {
            "status": "ok",
            "version": "0.1.0",
            "message": "API is running and ready to serve requests"
        }
    """
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        message=f"{settings.APP_NAME} is running and ready to serve requests",
    )
