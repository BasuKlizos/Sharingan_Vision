from fastapi import FastAPI

from app.logger import logger

from app.middleware.log_middleware import log_request_middleware

app = FastAPI(
    title="My FastAPI Application",
    description="A sample FastAPI application with structured logging and middleware.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.middleware("http")(log_request_middleware)


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")


@app.get("/ping")
async def ping():
    logger.debug("Ping endpoint called")
    return {"message": "pong"}