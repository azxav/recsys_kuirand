import json
from datetime import UTC, datetime

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from libs.common.config import get_settings
from libs.schemas.events import EventIn
from libs.schemas.recommend import SlateLogIn


def get_clickhouse_client() -> Client:
    s = get_settings()
    return clickhouse_connect.get_client(
        host=s.clickhouse_host,
        port=s.clickhouse_port,
        username=s.clickhouse_username,
        password=s.clickhouse_password,
        database=s.clickhouse_database,
    )


def _to_row(event: EventIn) -> list[object]:
    event_dt = datetime.fromtimestamp(event.ts_ms / 1000, tz=UTC).date()
    ctx_tab = event.context.tab if event.context else None
    model_versions = event.model_versions or {}
    experiment_assignments = event.experiment_assignments or {}
    return [
        event_dt,
        event.ts_ms,
        event.user_id,
        event.session_id or "",
        event.request_id or "",
        event.item_id,
        int(ctx_tab or 0),
        event.event_type,
        1 if event.event_type in {"click", "watch"} else 0,
        int(event.watch_time_ms or 0),
        int(event.item_duration_ms or 0),
        1 if (event.watch_time_ms or 0) > 10_000 else 0,
        1 if event.event_type == "like" else 0,
        1 if event.event_type == "follow" else 0,
        1 if event.event_type == "comment" else 0,
        1 if event.event_type == "forward" else 0,
        1 if event.event_type == "hate" else 0,
        1 if event.is_random_exposure else 0,
        model_versions.get("als", "unknown"),
        model_versions.get("blend_meta", "none"),
        event.served_rank,
        event.final_score,
        event.experiment_key or "",
        event.experiment_variant or "",
        json.dumps(experiment_assignments, sort_keys=True),
        json.dumps(model_versions, sort_keys=True),
    ]


def insert_events(events: list[EventIn]) -> None:
    if not events:
        return
    client = get_clickhouse_client()
    rows = [_to_row(event) for event in events]
    client.insert(
        "events_raw",
        rows,
        column_names=[
            "event_date",
            "ts_ms",
            "user_id",
            "session_id",
            "request_id",
            "item_id",
            "tab",
            "event_type",
            "is_click",
            "watch_time_ms",
            "item_duration_ms",
            "long_view",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "is_random_exposure",
            "retrieval_model_version",
            "ranker_model_version",
            "served_rank",
            "final_score",
            "experiment_key",
            "experiment_variant",
            "experiment_assignments_json",
            "model_versions_json",
        ],
    )


def _to_slate_row(slate: SlateLogIn) -> list[object]:
    event_dt = datetime.fromtimestamp(slate.ts_ms / 1000, tz=UTC).date()
    return [
        event_dt,
        slate.ts_ms,
        slate.request_id,
        slate.user_id,
        slate.candidate_count,
        slate.served_count,
        slate.fallback_used or "",
        json.dumps(slate.model_versions, sort_keys=True),
        json.dumps(slate.experiment_assignments, sort_keys=True),
        slate.item_ids,
        slate.item_scores,
        json.dumps(slate.latency_ms, sort_keys=True),
    ]


def insert_slates(slates: list[SlateLogIn]) -> None:
    if not slates:
        return
    client = get_clickhouse_client()
    rows = [_to_slate_row(slate) for slate in slates]
    client.insert(
        "recommend_slates_raw",
        rows,
        column_names=[
            "event_date",
            "ts_ms",
            "request_id",
            "user_id",
            "candidate_count",
            "served_count",
            "fallback_used",
            "model_versions_json",
            "experiment_assignments_json",
            "item_ids",
            "item_scores",
            "latency_json",
        ],
    )
