import asyncio
import json
import logging
import time

from prometheus_client import Counter, Gauge, start_http_server
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from libs.common.config import get_settings
from libs.common.logging import configure_logging
from libs.schemas.events import EventIn
from libs.schemas.recommend import SlateLogIn
from libs.storage.clickhouse import insert_events, insert_slates
from libs.storage.redis import get_redis_client, push_dlq

configure_logging()
logger = logging.getLogger(__name__)

WORKER_MESSAGES = Counter("recsys_worker_messages_total", "Messages consumed", ["stream", "status"])
WORKER_BATCH_SIZE = Gauge("recsys_worker_last_batch_size", "Last batch size")


async def _ensure_group(redis: Redis, stream_key: str, group_name: str) -> None:
    try:
        await redis.xgroup_create(stream_key, group_name, id="$", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def run_worker() -> None:
    s = get_settings()
    redis = await get_redis_client()

    await _ensure_group(redis, s.redis_stream_key, s.redis_stream_group)
    await _ensure_group(redis, s.redis_slate_stream_key, s.redis_stream_group)

    start_http_server(s.metrics_port)
    logger.info("worker started")

    while True:
        response = await redis.xreadgroup(
            groupname=s.redis_stream_group,
            consumername=s.redis_stream_consumer,
            streams={s.redis_stream_key: ">", s.redis_slate_stream_key: ">"},
            count=100,
            block=5000,
        )
        if not response:
            continue

        for stream_name, entries in response:
            WORKER_BATCH_SIZE.set(len(entries))
            stream_key = str(stream_name)

            if stream_key == s.redis_stream_key:
                valid: list[tuple[str, EventIn, str]] = []
                for msg_id, fields in entries:
                    raw_payload = fields.get("payload", "")
                    try:
                        payload = json.loads(raw_payload)
                        event = EventIn.model_validate(payload)
                        valid.append((str(msg_id), event, raw_payload))
                    except Exception as exc:
                        WORKER_MESSAGES.labels(stream_key, "invalid").inc()
                        await push_dlq(
                            source_stream=stream_key,
                            source_id=str(msg_id),
                            payload=raw_payload,
                            error=str(exc),
                            failed_at_ms=int(time.time() * 1000),
                        )
                        await redis.xack(stream_key, s.redis_stream_group, msg_id)

                if valid:
                    try:
                        insert_events([x[1] for x in valid])
                        await redis.xack(stream_key, s.redis_stream_group, *[x[0] for x in valid])
                        WORKER_MESSAGES.labels(stream_key, "ok").inc(len(valid))
                    except Exception as exc:
                        for msg_id, _, raw_payload in valid:
                            WORKER_MESSAGES.labels(stream_key, "dlq").inc()
                            await push_dlq(
                                source_stream=stream_key,
                                source_id=msg_id,
                                payload=raw_payload,
                                error=str(exc),
                                failed_at_ms=int(time.time() * 1000),
                            )
                            await redis.xack(stream_key, s.redis_stream_group, msg_id)

            elif stream_key == s.redis_slate_stream_key:
                valid_slates: list[tuple[str, SlateLogIn, str]] = []
                for msg_id, fields in entries:
                    raw_payload = fields.get("payload", "")
                    try:
                        payload = json.loads(raw_payload)
                        slate = SlateLogIn.model_validate(payload)
                        valid_slates.append((str(msg_id), slate, raw_payload))
                    except Exception as exc:
                        WORKER_MESSAGES.labels(stream_key, "invalid").inc()
                        await push_dlq(
                            source_stream=stream_key,
                            source_id=str(msg_id),
                            payload=raw_payload,
                            error=str(exc),
                            failed_at_ms=int(time.time() * 1000),
                        )
                        await redis.xack(stream_key, s.redis_stream_group, msg_id)

                if valid_slates:
                    try:
                        insert_slates([x[1] for x in valid_slates])
                        await redis.xack(
                            stream_key,
                            s.redis_stream_group,
                            *[x[0] for x in valid_slates],
                        )
                        WORKER_MESSAGES.labels(stream_key, "ok").inc(len(valid_slates))
                    except Exception as exc:
                        for msg_id, _, raw_payload in valid_slates:
                            WORKER_MESSAGES.labels(stream_key, "dlq").inc()
                            await push_dlq(
                                source_stream=stream_key,
                                source_id=msg_id,
                                payload=raw_payload,
                                error=str(exc),
                                failed_at_ms=int(time.time() * 1000),
                            )
                            await redis.xack(stream_key, s.redis_stream_group, msg_id)


if __name__ == "__main__":
    asyncio.run(run_worker())
