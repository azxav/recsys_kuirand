from typing import Literal

from pydantic import BaseModel


class DeviceContext(BaseModel):
    device_type: str | None = None
    locale: str | None = None
    tab: int | None = None


EventType = Literal[
    "impression",
    "click",
    "watch",
    "like",
    "follow",
    "comment",
    "forward",
    "hate",
]


class EventIn(BaseModel):
    user_id: int
    item_id: int
    ts_ms: int
    session_id: str | None = None
    event_type: EventType
    watch_time_ms: int | None = None
    item_duration_ms: int | None = None
    is_random_exposure: bool | None = None
    context: DeviceContext | None = None
    request_id: str | None = None
    served_rank: int | None = None
    final_score: float | None = None
    experiment_key: str | None = None
    experiment_variant: str | None = None
    experiment_assignments: dict[str, str] | None = None
    model_versions: dict[str, str] | None = None
