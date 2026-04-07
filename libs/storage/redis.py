import json
from typing import Any

from redis.asyncio import Redis

from libs.common.config import get_settings


async def get_redis_client() -> Redis:
    s = get_settings()
    return Redis.from_url(s.redis_url, decode_responses=True)


async def push_event(event: dict[str, Any]) -> str:
    s = get_settings()
    client = await get_redis_client()
    stream_id = await client.xadd(s.redis_stream_key, {"payload": json.dumps(event)})
    return str(stream_id)


async def push_events(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return []
    s = get_settings()
    client = await get_redis_client()
    pipe = client.pipeline()
    for event in events:
        pipe.xadd(s.redis_stream_key, {"payload": json.dumps(event)})
    out = await pipe.execute()
    return [str(x) for x in out]


async def push_slate_log(payload: dict[str, Any]) -> str:
    s = get_settings()
    client = await get_redis_client()
    stream_id = await client.xadd(s.redis_slate_stream_key, {"payload": json.dumps(payload)})
    return str(stream_id)


async def push_dlq(
    source_stream: str,
    source_id: str,
    payload: str,
    error: str,
    failed_at_ms: int,
) -> str:
    s = get_settings()
    client = await get_redis_client()
    stream_id = await client.xadd(
        s.redis_dlq_stream_key,
        {
            "source_stream": source_stream,
            "source_id": source_id,
            "payload": payload,
            "error": error,
            "failed_at_ms": str(failed_at_ms),
        },
    )
    return str(stream_id)


async def get_recent_seen(user_id: int, n: int) -> list[int]:
    client = await get_redis_client()
    items = await client.lrange(f"u:{user_id}:recent", 0, max(n - 1, 0))
    return [int(x) for x in items]


async def add_recent_seen(user_id: int, item_ids: list[int], ttl_seconds: int = 86400) -> None:
    if not item_ids:
        return
    client = await get_redis_client()
    key = f"u:{user_id}:recent"
    await client.lpush(key, *[str(i) for i in item_ids])
    await client.ltrim(key, 0, 999)
    await client.expire(key, ttl_seconds)


async def cache_slate(request_id: str, payload: dict[str, Any], ttl_seconds: int = 604800) -> None:
    client = await get_redis_client()
    await client.set(f"req:{request_id}:slate", json.dumps(payload), ex=ttl_seconds)


async def get_cached_experiment_assignment(experiment_key: str, user_id: int) -> str | None:
    client = await get_redis_client()
    return await client.get(f"u:{user_id}:exp:{experiment_key}")


async def set_cached_experiment_assignment(
    experiment_key: str,
    user_id: int,
    variant: str,
    ttl_seconds: int,
) -> None:
    client = await get_redis_client()
    await client.set(f"u:{user_id}:exp:{experiment_key}", variant, ex=ttl_seconds)
