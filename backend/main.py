"""FastAPI entry point with an isolated Phase 0 private-pilot profile.

``API_SERVING_PROFILE=phase0`` mounts only the operational health route and
the three Ticker Narratives read routes. It does not import the legacy product
routers and does not register prediction prewarming. ``full`` remains the
backward-compatible default for the legacy application.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.phase0 import routes as phase0_routes
from app.routes import health


logger = logging.getLogger("uvicorn.error")


async def startup_event():
    """Prewarm legacy experimental-signal artifacts in the full profile only."""

    from app.services.prediction_service import PredictionService

    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug mode: {settings.DEBUG}")
    try:
        prewarm_info = PredictionService.prewarm_model_artifacts()
        if prewarm_info.get("status") == "ready":
            logger.info(
                "Experimental signal artifacts preloaded: source=%s "
                "version=%s path=%s trained_at=%s",
                prewarm_info.get("artifact_source") or "unknown",
                prewarm_info.get("version") or "unknown",
                prewarm_info.get("artifact_path") or "unknown",
                prewarm_info.get("trained_at") or "unknown",
            )
        else:
            logger.warning(
                "Experimental signal artifact preload skipped: %s. "
                "Runtime signal loading will remain lazy.",
                prewarm_info.get("reason") or "artifact loader unavailable",
            )
    except Exception as exc:
        logger.warning(
            "Experimental signal artifact preload failed (%s: %s). "
            "Runtime signal loading will remain lazy.",
            type(exc).__name__,
            exc,
        )


async def shutdown_event():
    """Clean up resources on shutdown."""
    print("Shutting down API...")


def create_app(*, serving_profile: str | None = None) -> FastAPI:
    """Build the selected API surface without importing disabled products."""

    profile = serving_profile or settings.API_SERVING_PROFILE
    if profile not in {"full", "phase0"}:
        raise ValueError(f"unknown API serving profile: {profile!r}")

    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Financial narrative read API",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health.router)
    application.include_router(phase0_routes.router)

    if profile == "full":
        # These modules pull in the legacy sentiment, market and prediction
        # stack, so import them only for the explicitly selected full profile.
        from app.routes import dashboard, market, prediction, sentiment

        application.include_router(sentiment.router)
        application.include_router(prediction.router)
        application.include_router(market.router)
        application.include_router(dashboard.router)
        application.add_event_handler("startup", startup_event)
        application.add_event_handler("shutdown", shutdown_event)

    application.state.serving_profile = profile
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
