"""
AI Market Sentiment Dashboard - Backend API

Main FastAPI application entry point that aggregates data from all team members:
- Isaac: Data pipeline (raw social media posts + market data)
- Matthew: NLP sentiment analysis (FinBERT sentiment scores)
- Abhi: ML prediction model (stock movement predictions)
- Srish: Frontend (React dashboard served at /)
- Mihir: Backend API (integrates all components)

To run locally:
    cd backend
    PYTHONPATH=. python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    # API available at http://localhost:8000
    # Interactive docs at http://localhost:8000/docs
    # ReDoc documentation at http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import health, sentiment, prediction, market, dashboard

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Financial sentiment analysis and stock prediction API",
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
    TODO (Mihir + Abhi): Load pre-trained ML models into memory for fast inference
    TODO (Mihir): Verify connections to Isaac's data pipeline
    TODO (Mihir): Verify connections to Matthew's NLP module
    TODO (Mihir): Run health checks on dependent services
    TODO (Mihir): Load configuration from environment variables
    """
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug mode: {settings.DEBUG}")
    # Startup initialization code goes here


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
