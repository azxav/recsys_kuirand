CREATE TABLE IF NOT EXISTS experiments (
    experiment_key TEXT PRIMARY KEY,
    variants JSONB NOT NULL,
    start_at TIMESTAMPTZ NOT NULL,
    end_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'paused', 'ended')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_assignments (
    experiment_key TEXT NOT NULL REFERENCES experiments(experiment_key) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    variant TEXT NOT NULL,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (experiment_key, user_id)
);
