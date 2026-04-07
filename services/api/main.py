import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, make_asgi_app

from libs.common.logging import configure_logging
from libs.storage.vector_db import ensure_collection
from services.api.routers.events import router as events_router
from services.api.routers.experiments import router as experiments_router
from services.api.routers.health import router as health_router
from services.api.routers.recommend import router as recommend_router

configure_logging()
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "recsys_api_requests_total",
    "Total number of API requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "recsys_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        ensure_collection()
    except Exception as exc:
        logger.warning("qdrant init skipped: %s", exc)
    yield


app = FastAPI(title="KuaiRand RecSys API", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())

app.include_router(health_router)
app.include_router(events_router)
app.include_router(recommend_router)
app.include_router(experiments_router)


@app.middleware("http")
async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(elapsed)
    return response
