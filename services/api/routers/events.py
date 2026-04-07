from fastapi import APIRouter

from libs.schemas.events import EventIn
from libs.storage.redis import push_event

router = APIRouter(prefix="/v1/events", tags=["events"])


@router.post("")
async def ingest_event(event: EventIn) -> dict[str, str]:
    event_id = await push_event(event.model_dump(mode="json"))
    return {"status": "accepted", "stream_id": event_id}
