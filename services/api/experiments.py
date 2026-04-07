from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from libs.schemas.experiments import ExperimentResponse
from libs.storage.postgres import (
    experiment_cache_ttl_seconds,
    get_assignment,
    get_experiment,
    upsert_assignment,
    upsert_experiment,
)
from libs.storage.redis import get_cached_experiment_assignment, set_cached_experiment_assignment


def _normalize_variants(variants: dict[str, float]) -> list[tuple[str, float]]:
    total = float(sum(variants.values()))
    if total <= 0:
        raise ValueError("variant weights must sum > 0")
    return [(name, float(weight) / total) for name, weight in sorted(variants.items())]


def pick_variant(experiment_key: str, user_id: int, variants: dict[str, float]) -> str:
    normalized = _normalize_variants(variants)
    digest = hashlib.sha256(f"{experiment_key}:{user_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    acc = 0.0
    for variant, prob in normalized:
        acc += prob
        if bucket <= acc:
            return variant
    return normalized[-1][0]


def _experiment_is_active(exp: dict[str, Any]) -> bool:
    now = datetime.now(tz=UTC)
    start_at = exp["start_at"]
    end_at = exp["end_at"]
    status = str(exp["status"])
    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=UTC)
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=UTC)
    return status == "active" and start_at <= now <= end_at


async def assign_variant(experiment_key: str, user_id: int) -> tuple[str, str]:
    cached = await get_cached_experiment_assignment(experiment_key, user_id)
    if cached:
        return cached, "redis_cache"

    from_db = get_assignment(experiment_key, user_id)
    if from_db:
        exp = get_experiment(experiment_key)
        ttl = experiment_cache_ttl_seconds(exp["end_at"]) if exp else 3600
        await set_cached_experiment_assignment(experiment_key, user_id, from_db, ttl)
        return from_db, "postgres_assignment"

    exp = get_experiment(experiment_key)
    if not exp:
        raise ValueError(f"experiment {experiment_key} not found")
    if not _experiment_is_active(exp):
        raise ValueError(f"experiment {experiment_key} is not active")

    variant = pick_variant(experiment_key, user_id, exp["variants"])
    upsert_assignment(experiment_key, user_id, variant)
    await set_cached_experiment_assignment(
        experiment_key,
        user_id,
        variant,
        experiment_cache_ttl_seconds(exp["end_at"]),
    )
    return variant, "computed"


async def assign_variants(experiment_keys: list[str], user_id: int) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for key in experiment_keys:
        variant, _ = await assign_variant(key, user_id)
        assignments[key] = variant
    return assignments


def create_or_update_experiment(
    experiment_key: str,
    variants: dict[str, float],
    start_at: datetime,
    end_at: datetime,
    status: str,
) -> None:
    upsert_experiment(experiment_key, variants, start_at, end_at, status)


def fetch_experiment(experiment_key: str) -> ExperimentResponse | None:
    exp = get_experiment(experiment_key)
    if not exp:
        return None
    return ExperimentResponse(
        experiment_key=str(exp["experiment_key"]),
        variants={k: float(v) for k, v in dict(exp["variants"]).items()},
        start_at=exp["start_at"],
        end_at=exp["end_at"],
        status=str(exp["status"]),
    )
