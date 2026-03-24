from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.logger import logger
from app.middleware.log_middleware import log_request_middleware
from app.core.redis import redis_manager
from app.api.router import api_router
from app.core.config import settings    

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Redis...")
    await redis_manager.connect()

    yield

    logger.info("Disconnecting Redis...")
    await redis_manager.disconnect()


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware
app.middleware("http")(log_request_middleware)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/ping")
async def ping():
    logger.debug("Ping endpoint called")
    redis_client = await redis_manager.get_client()
    return {"message": "pong"}