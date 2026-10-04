.PHONY: lint type test api worker etl build-splits train-pop train-als train-bpr train-two-tower build-candidates train-ranker train-blend eval-ranker train-all smoke-train download-data eval smoke

lint:
	uv run ruff check libs services pipelines tests scripts

type:
	uv run mypy libs services pipelines

test:
	uv run pytest -m "not integration"

api:
	uv run uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	uv run python -m services.worker.main

etl:
	uv run python -m pipelines.spark.etl_events --input data/sample/events_sample.csv --output data/processed/events_processed.csv

build-splits:
	uv run python -m pipelines.training.build_time_splits --input data/processed/events_processed.csv --output-dir data/processed/splits

train-pop:
	uv run python -m pipelines.training.retrieval_popular --input data/processed/events_processed.csv --output artifacts/popular.json

train-als:
	uv run python -m pipelines.training.retrieval_implicit --input data/processed/events_processed.csv --output-dir artifacts/als_v2

train-bpr:
	uv run python -m pipelines.training.retrieval_bpr --input data/processed/events_processed.csv --output-dir artifacts/bpr_v1

train-two-tower:
	uv run python -m pipelines.training.retrieval_two_tower --input data/processed/events_processed.csv --output-dir artifacts/twotower_v1

build-candidates:
	uv run python -m pipelines.training.build_candidate_sets --interactions data/processed/events_processed.csv --als-dir artifacts/als_v2 --bpr-dir artifacts/bpr_v1 --two-tower-dir artifacts/twotower_v1 --output artifacts/ranker_v1/candidates.csv --candidate-k 500

train-ranker:
	uv run python -m pipelines.training.ranking_multi_objective --candidates artifacts/ranker_v1/candidates.csv --output-dir artifacts/ranker_v1

train-blend:
	uv run python -m pipelines.training.train_blend_meta --predictions artifacts/ranker_v1/objective_predictions.csv --output-model artifacts/ranker_v1/blend_model.cbm

eval-ranker:
	uv run python -m pipelines.training.eval_ranking --predictions artifacts/ranker_v1/objective_predictions.csv

train-all: train-als train-bpr train-two-tower build-candidates train-ranker train-blend

smoke-train: etl train-all eval-ranker

download-data:
	bash scripts/download_kuairand_pure.sh

eval:
	uv run python -m pipelines.training.offline_eval --data-dir data/raw/KuaiRand-Pure/data --output results/kuairand_pure_metrics.json

smoke:
	uv run python -m pipelines.training.offline_eval --smoke --output /tmp/kuairand-smoke-metrics.json
