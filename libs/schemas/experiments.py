from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ExperimentUpsertRequest(BaseModel):
    experiment_key: str = Field(min_length=1, max_length=128)
    variants: dict[str, float]
    start_at: datetime
    end_at: datetime
    status: str = Field(default="active")

    @model_validator(mode="after")
    def validate_payload(self) -> "ExperimentUpsertRequest":
        if not self.variants:
            raise ValueError("variants must not be empty")
        total = 0.0
        for variant, weight in self.variants.items():
            if not variant:
                raise ValueError("variant name must not be empty")
            if weight <= 0:
                raise ValueError("variant weights must be > 0")
            total += weight
        if total <= 0:
            raise ValueError("sum of weights must be > 0")
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be after start_at")
        if self.status not in {"active", "paused", "ended"}:
            raise ValueError("status must be active|paused|ended")
        return self


class ExperimentResponse(BaseModel):
    experiment_key: str
    variants: dict[str, float]
    start_at: datetime
    end_at: datetime
    status: str


class ExperimentAssignRequest(BaseModel):
    experiment_key: str = Field(min_length=1, max_length=128)
    user_id: int


class ExperimentAssignResponse(BaseModel):
    experiment_key: str
    user_id: int
    variant: str
    source: str
