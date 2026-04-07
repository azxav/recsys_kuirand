# KuaiRand RecSys MVP

Production-minded two-stage recommender MVP using `uv` + `pyproject.toml`.

## Stack
- Python 3.11+
- FastAPI
- Redis + Redis Streams
- ClickHouse
- Postgres
- Qdrant
- CatBoost + Implicit ALS/BPR + Two-Tower (PyTorch)
- MLflow
- Prometheus + Grafana

## Quickstart
```bash
uv sync --extra dev --extra spark --extra deep
cp .env.example .env
docker compose up -d
make api
```

## Common Commands
```bash
make lint
make type
make test
make api
make worker
make etl
make build-splits
make train-als
make train-bpr
make train-two-tower
make build-candidates
make train-ranker
make train-blend
make eval-ranker
make train-all
```

## API
- `POST /v1/events`
- `POST /v1/recommend`
- `GET /v1/items/{item_id}/similar`
- `POST /v1/experiments`
- `GET /v1/experiments/{experiment_key}`
- `POST /v1/experiments/assign`
- `GET /health`
- `GET /ready`
- `GET /metrics`

`/v1/recommend` returns:
- `final_score`
- `objective_scores` (`click`, `long_view`, `watch_time`)
- `blend_features` (`als_score`, `bpr_score`, `two_tower_score`)
- `candidate_count`
- `experiment_assignments`

## Notes
- `/ready` now runs dependency checks and returns HTTP `503` when required components are unavailable.
- Recommend requests auto-log impression events and served slates through Redis Streams for worker ingestion into ClickHouse.
- Prometheus scrapes API and worker metrics; Grafana dashboard is provisioned at startup.
