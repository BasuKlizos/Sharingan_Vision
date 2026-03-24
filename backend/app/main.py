from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.logger import logger
from app.middleware.log_middleware import log_request_middleware
from app.core.redis import redis_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Redis...")
    await redis_manager.connect()

    yield

    logger.info("Disconnecting Redis...")
    await redis_manager.disconnect()


app = FastAPI(
    title="My FastAPI Application",
    description="Real-time WebRTC-based AI proctoring backend with FastAPI and computer vision.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # frontend Origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log Request Middleware
app.middleware("http")(log_request_middleware)


@app.get("/ping")
async def ping():
    """
    Ping endpoint to check if the application is alive.

    Returns:
        dict: {"message": "pong"}
    """
    logger.debug("Ping endpoint called")
    return {"message": "pong"}