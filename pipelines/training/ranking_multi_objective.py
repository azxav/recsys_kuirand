import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from libs.common.config import get_settings

FEATURE_COLS = [
    "als_score",
    "bpr_score",
    "two_tower_score",
    "max_score",
    "min_score",
    "tab",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["two_tower_score"] = out.get("two_tower_score", 0.0)
    out["max_score"] = out[["als_score", "bpr_score", "two_tower_score"]].max(axis=1)
    out["min_score"] = out[["als_score", "bpr_score", "two_tower_score"]].min(axis=1)
    out["tab"] = out["tab"].fillna(0)
    out["watch_time_log"] = np.log1p(out["watch_time_ms"].clip(lower=0))
    return out


def run(candidate_path: str, output_dir: str) -> None:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(f"{s.mlflow_experiment_prefix}-ranking")

    df = pd.read_csv(candidate_path)
    df = _build_features(df)

    X = df[FEATURE_COLS]
    y_click = df["is_click"].astype(int)
    y_long = df["long_view"].astype(int)
    y_watch = df["watch_time_log"].astype(float)

    click_model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=250,
        depth=6,
        learning_rate=0.08,
        verbose=False,
        random_seed=42,
    )
    long_model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=250,
        depth=6,
        learning_rate=0.08,
        verbose=False,
        random_seed=42,
    )
    watch_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=350,
        depth=6,
        learning_rate=0.06,
        verbose=False,
        random_seed=42,
    )

    with mlflow.start_run(run_name="ranking_multi_objective"):
        mlflow.log_param("feature_cols", ",".join(FEATURE_COLS))
        mlflow.log_param("rows", int(len(df)))

        click_model.fit(X, y_click)
        long_model.fit(X, y_long)
        watch_model.fit(X, y_watch)

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        click_model.save_model(str(outdir / "click_model.cbm"))
        long_model.save_model(str(outdir / "long_view_model.cbm"))
        watch_model.save_model(str(outdir / "watch_time_model.cbm"))

        df_preds = df[
            [
                "query_id",
                "user_id",
                "item_id",
                "is_click",
                "long_view",
                "watch_time_ms",
                "tab",
            ]
        ].copy()
        df_preds["pred_click"] = click_model.predict_proba(X)[:, 1]
        df_preds["pred_long_view"] = long_model.predict_proba(X)[:, 1]
        df_preds["pred_watch_time"] = watch_model.predict(X)
        df_preds["als_score"] = df["als_score"]
        df_preds["bpr_score"] = df["bpr_score"]
        df_preds["two_tower_score"] = df["two_tower_score"]
        out_path = outdir / "objective_predictions.csv"
        df_preds.to_csv(out_path, index=False)

        mlflow.log_artifact(str(outdir / "click_model.cbm"))
        mlflow.log_artifact(str(outdir / "long_view_model.cbm"))
        mlflow.log_artifact(str(outdir / "watch_time_model.cbm"))
        mlflow.log_artifact(str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(candidate_path=args.candidates, output_dir=args.output_dir)
