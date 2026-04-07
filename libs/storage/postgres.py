from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from libs.common.config import get_settings


def _connect() -> psycopg.Connection[Any]:
    s = get_settings()
    return psycopg.connect(s.postgres_dsn, row_factory=dict_row)


def upsert_experiment(
    experiment_key: str,
    variants: dict[str, float],
    start_at: datetime,
    end_at: datetime,
    status: str,
) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiments (
                    experiment_key, variants, start_at, end_at, status
                )
                VALUES (%s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (experiment_key) DO UPDATE
                SET variants = EXCLUDED.variants,
                    start_at = EXCLUDED.start_at,
                    end_at = EXCLUDED.end_at,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                """,
                (experiment_key, json.dumps(variants), start_at, end_at, status),
            )


def get_experiment(experiment_key: str) -> dict[str, Any] | None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT experiment_key, variants, start_at, end_at, status
                FROM experiments
                WHERE experiment_key = %s
                """,
                (experiment_key,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_assignment(experiment_key: str, user_id: int) -> str | None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT variant
                FROM experiment_assignments
                WHERE experiment_key = %s AND user_id = %s
                """,
                (experiment_key, user_id),
            )
            row = cur.fetchone()
            return str(row["variant"]) if row else None


def upsert_assignment(experiment_key: str, user_id: int, variant: str) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_assignments (experiment_key, user_id, variant)
                VALUES (%s, %s, %s)
                ON CONFLICT (experiment_key, user_id) DO UPDATE
                SET variant = EXCLUDED.variant
                """,
                (experiment_key, user_id, variant),
            )


def experiment_cache_ttl_seconds(end_at: datetime) -> int:
    now = datetime.now(tz=UTC)
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=UTC)
    ttl = int((end_at - now).total_seconds())
    return max(ttl, 300)


def ping_postgres() -> bool:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()
            return row is not None
