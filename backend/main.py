"""
AI Market Sentiment Dashboard - Backend API

Main FastAPI application entry point that aggregates data from all team members:
- Isaac: Data pipeline (raw social media posts + market data)
- Matthew: NLP sentiment analysis (FinBERT sentiment scores)
- Abhi: Experimental market signal artifacts
- Srish: Frontend (React dashboard served at /)
- Mihir: Backend API (integrates all components)

To run locally:
    cd backend
    PYTHONPATH=. python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    # API available at http://localhost:8000
    # Interactive docs at http://localhost:8000/docs
    # ReDoc documentation at http://localhost:8000/redoc
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import health, sentiment, prediction, market, dashboard
from app.services.prediction_service import PredictionService


logger = logging.getLogger("uvicorn.error")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Financial sentiment analysis and experimental market signal API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO (Mihir): Restrict to frontend domain in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes from different modules
app.include_router(health.router)
app.include_router(sentiment.router)
app.include_router(prediction.router)
app.include_router(market.router)
app.include_router(dashboard.router)


@app.on_event("startup")
async def startup_event():
    """
    Initialize app on startup.
    
    TODO (Mihir): Initialize database connections (if using database)
    TODO (Mihir + Abhi): Load validated market signal artifacts into memory for fast inference
    TODO (Mihir): Verify connections to Isaac's data pipeline
    TODO (Mihir): Verify connections to Matthew's NLP module
    TODO (Mihir): Run health checks on dependent services
    TODO (Mihir): Load configuration from environment variables
    """
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug mode: {settings.DEBUG}")
    try:
        prewarm_info = PredictionService.prewarm_model_artifacts()
        if prewarm_info.get("status") == "ready":
            logger.info(
                "Experimental signal artifacts preloaded: source=%s version=%s path=%s trained_at=%s",
                prewarm_info.get("artifact_source") or "unknown",
                prewarm_info.get("version") or "unknown",
                prewarm_info.get("artifact_path") or "unknown",
                prewarm_info.get("trained_at") or "unknown",
            )
        else:
            logger.warning(
                "Experimental signal artifact preload skipped: %s. Runtime signal loading will remain lazy.",
                prewarm_info.get("reason") or "artifact loader unavailable",
            )
    except Exception as exc:
        logger.warning(
            "Experimental signal artifact preload failed (%s: %s). Runtime signal loading will remain lazy.",
            type(exc).__name__,
            exc,
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("Shutting down API...")
    # TODO: Close database connections
    # TODO: Clean up ML models


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
