from fastapi import APIRouter, HTTPException

from libs.schemas.experiments import (
    ExperimentAssignRequest,
    ExperimentAssignResponse,
    ExperimentResponse,
    ExperimentUpsertRequest,
)
from services.api.experiments import (
    assign_variant,
    create_or_update_experiment,
    fetch_experiment,
)

router = APIRouter(prefix="/v1/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse)
async def upsert_experiment(req: ExperimentUpsertRequest) -> ExperimentResponse:
    create_or_update_experiment(
        experiment_key=req.experiment_key,
        variants=req.variants,
        start_at=req.start_at,
        end_at=req.end_at,
        status=req.status,
    )
    out = fetch_experiment(req.experiment_key)
    if out is None:
        raise HTTPException(status_code=500, detail="failed to persist experiment")
    return out


@router.get("/{experiment_key}", response_model=ExperimentResponse)
async def get_experiment(experiment_key: str) -> ExperimentResponse:
    out = fetch_experiment(experiment_key)
    if out is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return out


@router.post("/assign", response_model=ExperimentAssignResponse)
async def assign(req: ExperimentAssignRequest) -> ExperimentAssignResponse:
    try:
        variant, source = await assign_variant(req.experiment_key, req.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExperimentAssignResponse(
        experiment_key=req.experiment_key,
        user_id=req.user_id,
        variant=variant,
        source=source,
    )
