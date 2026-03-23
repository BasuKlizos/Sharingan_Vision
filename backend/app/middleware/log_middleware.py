import http
import time
import uuid

from fastapi import Request, Response
from app.logger import logger


async def log_request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    url = str(request.url)
    host = getattr(getattr(request, "client", None), "host", "-")

    start = time.perf_counter()

    try:
        response: Response = await call_next(request)
    except Exception:
        logger.exception(
            f"[{request_id}] {host} - \"{request.method} {url}\" 500 Internal Server Error",
            extra={"hide_src": True},
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    status_phrase = http.HTTPStatus(response.status_code).phrase

    logger.info(
        f"[{request_id}] {host} - \"{request.method} {url}\" "
        f"{response.status_code} {status_phrase} {elapsed_ms:.2f} ms",
        extra={"hide_src": True},
    )

    return response