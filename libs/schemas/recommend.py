from pydantic import BaseModel, Field

from libs.schemas.events import DeviceContext


class RecommendRequest(BaseModel):
    user_id: int
    session_id: str | None = None
    k: int = Field(default=20, ge=1, le=200)
    tab: int | None = None
    device: DeviceContext | None = None
    forbid_seen_last_n: int = Field(default=200, ge=0, le=5000)
    candidate_k: int = Field(default=500, ge=20, le=5000)
    experiment_keys: list[str] = Field(default_factory=list, max_length=20)
    debug: bool = False


class RecommendedItem(BaseModel):
    item_id: int
    rank: int
    final_score: float
    objective_scores: dict[str, float]
    blend_features: dict[str, float]


class RecommendResponse(BaseModel):
    request_id: str
    user_id: int
    items: list[RecommendedItem]
    model_versions: dict[str, str]
    latency_ms: dict[str, float]
    fallback_used: str | None = None
    candidate_count: int
    experiment_assignments: dict[str, str] = Field(default_factory=dict)


class SlateLogIn(BaseModel):
    request_id: str
    user_id: int
    ts_ms: int
    candidate_count: int
    served_count: int
    fallback_used: str | None = None
    model_versions: dict[str, str]
    experiment_assignments: dict[str, str] = Field(default_factory=dict)
    item_ids: list[int]
    item_scores: list[float]
    latency_ms: dict[str, float]
