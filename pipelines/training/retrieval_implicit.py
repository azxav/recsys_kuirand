import argparse
import json
from pathlib import Path

import implicit
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from libs.common.config import get_settings


def _build_matrix(df: pd.DataFrame) -> tuple[coo_matrix, dict[int, int], dict[int, int]]:
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    rows = df["item_id"].map(item2idx).astype(int).to_numpy()
    cols = df["user_id"].map(user2idx).astype(int).to_numpy()
    data = (df["is_click"].fillna(0).astype(float).to_numpy() + 0.1).astype(np.float32)
    mat = coo_matrix((data, (rows, cols)), shape=(len(items), len(users)))
    return mat, user2idx, item2idx


def recall_at_k(model: implicit.als.AlternatingLeastSquares, mat: coo_matrix, k: int = 10) -> float:
    csr = mat.tocsr().T
    total = 0
    hit = 0
    for user_idx in range(csr.shape[0]):
        liked = csr[user_idx].indices
        if liked.size == 0:
            continue
        recs, _ = model.recommend(user_idx, csr[user_idx], N=k, filter_already_liked_items=False)
        total += 1
        if any(i in liked for i in recs):
            hit += 1
    return hit / max(total, 1)


def run(input_path: str, output_dir: str, factors: int = 64, iterations: int = 20) -> None:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(f"{s.mlflow_experiment_prefix}-retrieval")

    df = pd.read_csv(input_path)
    if "item_id" not in df.columns:
        df = df.rename(columns={"video_id": "item_id"})

    mat, user2idx, item2idx = _build_matrix(df)
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        random_state=42,
    )

    with mlflow.start_run(run_name="als"):
        mlflow.log_param("factors", factors)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("rows", int(len(df)))

        model.fit(mat)

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "item_factors.npy", model.item_factors)
        np.save(outdir / "user_factors.npy", model.user_factors)

        idx2item = {v: k for k, v in item2idx.items()}
        mapping = {str(i): int(idx2item[i]) for i in range(len(idx2item))}
        with open(outdir / "item_index.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        with open(outdir / "user_index.json", "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in user2idx.items()}, f, indent=2)

        metrics = {"recall_at_10": recall_at_k(model, mat, k=10)}
        with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_metric("recall_at_10", float(metrics["recall_at_10"]))
        mlflow.log_artifact(str(outdir / "item_factors.npy"))
        mlflow.log_artifact(str(outdir / "user_factors.npy"))
        mlflow.log_artifact(str(outdir / "item_index.json"))
        mlflow.log_artifact(str(outdir / "user_index.json"))
        mlflow.log_artifact(str(outdir / "metrics.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run(args.input, args.output_dir, factors=args.factors, iterations=args.iterations)
