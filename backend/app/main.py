from fastapi import FastAPI
from contextlib import asynccontextmanager

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
    description="A sample FastAPI application with structured logging and middleware.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware
app.middleware("http")(log_request_middleware)


@app.get("/ping")
async def ping():
    logger.debug("Ping endpoint called")
    return {"message": "pong"}