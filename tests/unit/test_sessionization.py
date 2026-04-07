import pandas as pd

from pipelines.spark.etl_events import sessionize


def test_sessionize_30_min_gap() -> None:
    df = pd.DataFrame(
        [
            {"user_id": 1, "time_ms": 1_000_000},
            {"user_id": 1, "time_ms": 1_000_000 + 29 * 60 * 1000},
            {"user_id": 1, "time_ms": 1_000_000 + 61 * 60 * 1000},
        ]
    )
    out = sessionize(df)
    assert out.iloc[0]["session_id"] == "1-1"
    assert out.iloc[1]["session_id"] == "1-1"
    assert out.iloc[2]["session_id"] == "1-2"
