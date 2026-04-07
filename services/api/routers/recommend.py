from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter

from libs.common.config import get_settings
from libs.schemas.recommend import RecommendedItem, RecommendRequest, RecommendResponse
from libs.storage.redis import (
    add_recent_seen,
    cache_slate,
    get_recent_seen,
    push_events,
    push_slate_log,
)
from libs.storage.vector_db import query_similar
from services.api.experiments import assign_variants
from services.api.modeling import RankedCandidate, merge_candidates, rank_candidates

router = APIRouter(prefix="/v1", tags=["recommend"])


def _fallback_candidates(k: int) -> list[RankedCandidate]:
    return [
        {
            "item_id": 1000 + i,
            "final_score": 1.0 / (i + 1),
            "objective_scores": {
                "click": 0.0,
                "long_view": 0.0,
                "watch_time": 0.0,
            },
            "blend_features": {
                "als_score": 0.0,
                "bpr_score": 0.0,
                "two_tower_score": 0.0,
            },
        }
        for i in range(k)
    ]


def _model_versions() -> dict[str, str]:
    return {
        "als": "als_v2",
        "bpr": "bpr_v1",
        "two_tower": "twotower_v1",
        "rank_click": "ranker_v1",
        "rank_long_view": "ranker_v1",
        "rank_watch_time": "ranker_v1",
        "blend_meta": "ranker_v1",
    }


def _primary_experiment(assignments: dict[str, str]) -> tuple[str | None, str | None]:
    if not assignments:
        return None, None
    key = sorted(assignments.keys())[0]
    return key, assignments[key]


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    settings = get_settings()
    request_id = str(uuid.uuid4())
    model_versions = _model_versions()

    assignments: dict[str, str] = {}
    fallback_used: str | None = None

    if req.experiment_keys:
        try:
            assignments = await assign_variants(req.experiment_keys, req.user_id)
        except Exception:
            assignments = {}

    t_feat0 = time.perf_counter()
    try:
        seen = set(
            await asyncio.wait_for(
                get_recent_seen(req.user_id, req.forbid_seen_last_n),
                timeout=settings.recommend_feature_timeout_ms / 1000,
            )
        )
    except Exception:
        seen = set()
        fallback_used = "feature_timeout"
    t_feat1 = time.perf_counter()

    t_ret0 = time.perf_counter()
    candidate_k = req.candidate_k or settings.candidate_default_k
    try:
        merged_candidates = await asyncio.wait_for(
            asyncio.to_thread(merge_candidates, req.user_id, candidate_k),
            timeout=settings.recommend_retrieval_timeout_ms / 1000,
        )
    except Exception:
        merged_candidates = []
        fallback_used = "retrieval_timeout"

    if not merged_candidates and fallback_used is None:
        fallback_used = "popular"
    t_ret1 = time.perf_counter()

    t_rank0 = time.perf_counter()
    try:
        scored = await asyncio.wait_for(
            asyncio.to_thread(rank_candidates, merged_candidates, req.tab),
            timeout=settings.recommend_ranking_timeout_ms / 1000,
        )
    except Exception:
        scored = []
        fallback_used = "ranking_timeout"

    if not scored:
        scored = _fallback_candidates(req.k)

    filtered = [row for row in scored if row["item_id"] not in seen]
    if not filtered:
        filtered = _fallback_candidates(req.k)
        fallback_used = "popular_after_seen_filter"

    filtered.sort(key=lambda x: x["final_score"], reverse=True)
    topk = filtered[: req.k]
    items = [
        RecommendedItem(
            item_id=row["item_id"],
            rank=rank,
            final_score=row["final_score"],
            objective_scores={
                "click": row["objective_scores"]["click"],
                "long_view": row["objective_scores"]["long_view"],
                "watch_time": row["objective_scores"]["watch_time"],
            },
            blend_features={
                "als_score": row["blend_features"]["als_score"],
                "bpr_score": row["blend_features"]["bpr_score"],
                "two_tower_score": row["blend_features"].get("two_tower_score", 0.0),
            },
        )
        for rank, row in enumerate(topk, start=1)
    ]
    t_rank1 = time.perf_counter()

    await add_recent_seen(req.user_id, [x.item_id for x in items])

    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    exp_key, exp_variant = _primary_experiment(assignments)

    if settings.auto_log_impressions and items:
        impressions = [
            {
                "user_id": req.user_id,
                "item_id": item.item_id,
                "ts_ms": now_ms,
                "session_id": req.session_id,
                "event_type": "impression",
                "watch_time_ms": None,
                "item_duration_ms": None,
                "is_random_exposure": None,
                "context": {"tab": req.tab},
                "request_id": request_id,
                "served_rank": item.rank,
                "final_score": item.final_score,
                "experiment_key": exp_key,
                "experiment_variant": exp_variant,
                "experiment_assignments": assignments,
                "model_versions": model_versions,
            }
            for item in items
        ]
        try:
            await push_events(impressions)
        except Exception:
            pass

    payload = {
        "user_id": req.user_id,
        "items": [x.model_dump() for x in items],
        "model_versions": model_versions,
        "experiment_assignments": assignments,
    }
    await cache_slate(request_id, payload)

    t1 = time.perf_counter()
    latency = {
        "feature_fetch": (t_feat1 - t_feat0) * 1000,
        "retrieval": (t_ret1 - t_ret0) * 1000,
        "ranking": (t_rank1 - t_rank0) * 1000,
        "total": (t1 - t0) * 1000,
    }

    if settings.auto_log_slates:
        try:
            await push_slate_log(
                {
                    "request_id": request_id,
                    "user_id": req.user_id,
                    "ts_ms": now_ms,
                    "candidate_count": len(merged_candidates),
                    "served_count": len(items),
                    "fallback_used": fallback_used,
                    "model_versions": model_versions,
                    "experiment_assignments": assignments,
                    "item_ids": [x.item_id for x in items],
                    "item_scores": [x.final_score for x in items],
                    "latency_ms": latency,
                }
            )
        except Exception:
            pass

    return RecommendResponse(
        request_id=request_id,
        user_id=req.user_id,
        items=items,
        model_versions=model_versions,
        latency_ms=latency,
        fallback_used=fallback_used,
        candidate_count=len(merged_candidates),
        experiment_assignments=assignments,
    )


@router.get("/items/{item_id}/similar")
async def similar_items(item_id: int, k: int = 20) -> dict[str, list[dict[str, float | int]]]:
    sims = query_similar(item_id, limit=k)
    return {"items": [{"item_id": i, "score": s} for i, s in sims]}
