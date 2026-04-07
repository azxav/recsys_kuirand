from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="kuairand-recsys", alias="APP_NAME")
    env: str = Field(default="dev", alias="ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    metrics_host: str = Field(default="0.0.0.0", alias="METRICS_HOST")
    metrics_port: int = Field(default=9001, alias="METRICS_PORT")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_stream_key: str = Field(default="events_stream", alias="REDIS_STREAM_KEY")
    redis_stream_group: str = Field(default="events_workers", alias="REDIS_STREAM_GROUP")
    redis_stream_consumer: str = Field(default="worker-1", alias="REDIS_STREAM_CONSUMER")

    clickhouse_host: str = Field(default="localhost", alias="CLICKHOUSE_HOST")
    clickhouse_port: int = Field(default=8123, alias="CLICKHOUSE_PORT")
    clickhouse_username: str = Field(default="default", alias="CLICKHOUSE_USERNAME")
    clickhouse_password: str = Field(default="", alias="CLICKHOUSE_PASSWORD")
    clickhouse_database: str = Field(default="default", alias="CLICKHOUSE_DATABASE")

    postgres_dsn: str = Field(
        default="postgresql://recsys:recsys@localhost:5432/recsys",
        alias="POSTGRES_DSN",
    )

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="items_v1", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=128, alias="QDRANT_VECTOR_SIZE")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    model_version: str = Field(default="retrieval-pop-v1", alias="MODEL_VERSION")
    als_artifact_dir: str = Field(default="artifacts/als_v2", alias="ALS_ARTIFACT_DIR")
    bpr_artifact_dir: str = Field(default="artifacts/bpr_v1", alias="BPR_ARTIFACT_DIR")
    two_tower_artifact_dir: str = Field(
        default="artifacts/twotower_v1",
        alias="TWO_TOWER_ARTIFACT_DIR",
    )
    ranker_artifact_dir: str = Field(default="artifacts/ranker_v1", alias="RANKER_ARTIFACT_DIR")
    blend_model_path: str = Field(
        default="artifacts/ranker_v1/blend_model.cbm",
        alias="BLEND_MODEL_PATH",
    )
    candidate_default_k: int = Field(default=500, alias="CANDIDATE_DEFAULT_K")
    enable_two_tower: bool = Field(default=True, alias="ENABLE_TWO_TOWER")
    als_weight: float = Field(default=1.0, alias="ALS_WEIGHT")
    bpr_weight: float = Field(default=1.0, alias="BPR_WEIGHT")
    two_tower_weight: float = Field(default=1.0, alias="TWO_TOWER_WEIGHT")

    redis_slate_stream_key: str = Field(default="slate_stream", alias="REDIS_SLATE_STREAM_KEY")
    redis_dlq_stream_key: str = Field(default="events_dlq", alias="REDIS_DLQ_STREAM_KEY")

    recommend_feature_timeout_ms: int = Field(default=30, alias="RECOMMEND_FEATURE_TIMEOUT_MS")
    recommend_retrieval_timeout_ms: int = Field(default=60, alias="RECOMMEND_RETRIEVAL_TIMEOUT_MS")
    recommend_ranking_timeout_ms: int = Field(default=60, alias="RECOMMEND_RANKING_TIMEOUT_MS")
    recommend_total_timeout_ms: int = Field(default=150, alias="RECOMMEND_TOTAL_TIMEOUT_MS")
    auto_log_impressions: bool = Field(default=True, alias="AUTO_LOG_IMPRESSIONS")
    auto_log_slates: bool = Field(default=True, alias="AUTO_LOG_SLATES")

    mlflow_experiment_prefix: str = Field(
        default="kuairand-recsys",
        alias="MLFLOW_EXPERIMENT_PREFIX",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
