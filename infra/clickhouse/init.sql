CREATE TABLE IF NOT EXISTS events_raw
(
    event_date Date,
    ts_ms UInt64,
    user_id UInt32,
    session_id String,
    request_id String,
    item_id UInt32,
    tab UInt8,
    event_type LowCardinality(String),
    is_click UInt8,
    watch_time_ms UInt32,
    item_duration_ms UInt32,
    long_view UInt8,
    is_like UInt8,
    is_follow UInt8,
    is_comment UInt8,
    is_forward UInt8,
    is_hate UInt8,
    is_random_exposure UInt8,
    retrieval_model_version String,
    ranker_model_version String,
    served_rank Nullable(UInt16),
    final_score Nullable(Float32),
    experiment_key String,
    experiment_variant String,
    experiment_assignments_json String,
    model_versions_json String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_date, user_id, session_id, ts_ms);

CREATE TABLE IF NOT EXISTS recommend_slates_raw
(
    event_date Date,
    ts_ms UInt64,
    request_id String,
    user_id UInt32,
    candidate_count UInt32,
    served_count UInt16,
    fallback_used String,
    model_versions_json String,
    experiment_assignments_json String,
    item_ids Array(UInt32),
    item_scores Array(Float32),
    latency_json String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_date, user_id, ts_ms);
