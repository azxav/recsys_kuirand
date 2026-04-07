import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from libs.common.config import get_settings

FEATURES = [
    "pred_click",
    "pred_long_view",
    "pred_watch_time",
    "als_score",
    "bpr_score",
    "two_tower_score",
    "tab",
]


def run(predictions_path: str, output_model_path: str) -> None:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(f"{s.mlflow_experiment_prefix}-blend")

    df = pd.read_csv(predictions_path)
    if "two_tower_score" not in df.columns:
        df["two_tower_score"] = 0.0

    target = (
        0.35 * df["is_click"].astype(float)
        + 0.35 * df["long_view"].astype(float)
        + 0.30 * np.log1p(df["watch_time_ms"].astype(float).clip(lower=0))
    )

    X = df[FEATURES]
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=350,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        random_seed=42,
    )

    with mlflow.start_run(run_name="blend_meta"):
        mlflow.log_param("rows", int(len(df)))
        mlflow.log_param("feature_cols", ",".join(FEATURES))
        model.fit(X, target)

        output_path = Path(output_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_path))

        mlflow.log_artifact(str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-model", required=True)
    args = parser.parse_args()
    run(predictions_path=args.predictions, output_model_path=args.output_model)
