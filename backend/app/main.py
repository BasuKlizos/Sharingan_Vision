from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.logger import logger
from app.middleware.log_middleware import log_request_middleware
from app.core.redis import redis_manager
from app.core.mongodb import mongo_manager
from app.modules.proctoring.flush_service import get_flush_service
from app.api.monitoring import router as monitoring_router
from app.api.router import api_router
from app.core.config import settings    

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Redis...")
    await redis_manager.connect()

    try:
        logger.info("Connecting to MongoDB...")
        await mongo_manager.connect()
    except Exception as exc:
        # Keep the app running even if Mongo isn't available (Redis-only mode).
        logger.warning("MongoDB connection failed (continuing) | error=%s", exc)

    # Start proctoring flush service
    try:
        flush_service = get_flush_service()
        await flush_service.start()
    except Exception as exc:
        logger.error("Failed to start proctoring flush service | error=%s", exc)

    yield

    # Stop proctoring flush service
    try:
        flush_service = get_flush_service()
        await flush_service.stop()
    except Exception as exc:
        logger.error("Failed to stop proctoring flush service | error=%s", exc)

    logger.info("Disconnecting Redis...")
    await redis_manager.disconnect()

    try:
        logger.info("Disconnecting MongoDB...")
        await mongo_manager.disconnect()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"], # frontend Origin
    allow_origins=["*"], # frontend Origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log Request Middleware
app.middleware("http")(log_request_middleware)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(monitoring_router, prefix="/api", tags=["Gaze Monitoring"])

@app.get("/ping")
async def ping():
    """
    Ping endpoint to check if the application is alive.

    Returns:
        dict: {"message": "pong"}
    """
    redis_client = await redis_manager.get_client()
    return {"message": "pong"}


@app.get("/api/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    Returns status 'healthy' if the API is operational
    """
    return {"status": "healthy", "version": settings.VERSION}
