"""
Health Check Route

Simple endpoint to verify the API is running and operational.
"""

from fastapi import APIRouter
from app.config import settings
from app.models.schemas import HealthCheckResponse

router = APIRouter()


@router.get("/test", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthCheckResponse: API status and version information

    Example:
        GET /test
        Response: {
            "status": "ok",
            "version": "0.1.0",
            "message": "API is running and ready to serve requests"
        }
    """
    return HealthCheckResponse(
        status="ok",
        version=settings.APP_VERSION,
        message=f"{settings.APP_NAME} is running and ready to serve requests",
    )
