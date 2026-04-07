from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from libs.storage.clickhouse import get_clickhouse_client
from libs.storage.postgres import ping_postgres
from libs.storage.redis import get_redis_client
from libs.storage.vector_db import get_qdrant_client
from services.api.modeling import get_ranker_store, get_retrieval_store

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, Any]:
    checks: dict[str, bool] = {}

    try:
        redis = await get_redis_client()
        checks["redis"] = bool(await redis.ping())
    except Exception:
        checks["redis"] = False

    try:
        ch = get_clickhouse_client()
        ch.query("SELECT 1")
        checks["clickhouse"] = True
    except Exception:
        checks["clickhouse"] = False

    try:
        qdrant = get_qdrant_client()
        qdrant.get_collections()
        checks["qdrant"] = True
    except Exception:
        checks["qdrant"] = False

    try:
        checks["postgres"] = ping_postgres()
    except Exception:
        checks["postgres"] = False

    try:
        retrieval = get_retrieval_store()
        checks["retrieval_store"] = bool(
            retrieval.als_item_factors is not None or retrieval.bpr_item_factors is not None
        )
    except Exception:
        checks["retrieval_store"] = False

    try:
        ranker = get_ranker_store()
        checks["ranker_store"] = ranker.ready()
    except Exception:
        checks["ranker_store"] = False

    required = ["redis", "clickhouse", "qdrant", "postgres"]
    ok = all(checks.get(k, False) for k in required)
    payload = {"status": "ready" if ok else "not_ready", "checks": checks}
    if not ok:
        raise HTTPException(status_code=503, detail=payload)
    return payload
