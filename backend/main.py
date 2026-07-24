"""FastAPI entry point for the Phase 0 Ticker Narratives read API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.phase0 import routes as phase0_routes
from app.routes import health


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log startup without loading retired prediction or sentiment artifacts."""
    print("Starting Ticker Narratives API v1.0.0-phase0")
    print(f"Debug mode: {settings.DEBUG}")
    yield
    print("Shutting down Ticker Narratives API...")

# Initialize FastAPI app
app = FastAPI(
    title="Ticker Narratives API",
    version="1.0.0-phase0",
    description="Read API for cited coverage themes across the Phase 0 ticker universe.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The legacy sentiment, prediction, market, and dashboard routes are intentionally
# not mounted in Phase 0. The product surface is limited to cited ticker narratives.
app.include_router(health.router)
app.include_router(phase0_routes.router)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
