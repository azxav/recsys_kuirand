import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from libs.common.config import get_settings


def _build_interactions(
    df: pd.DataFrame,
) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
    users = sorted(df["user_id"].unique().tolist())
    items = sorted(df["item_id"].unique().tolist())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    interactions = [
        (user2idx[int(row.user_id)], item2idx[int(row.item_id)])
        for row in df.itertuples(index=False)
    ]
    return interactions, user2idx, item2idx


def run(
    input_path: str,
    output_dir: str,
    factors: int = 64,
    epochs: int = 8,
    batch_size: int = 2048,
    lr: float = 0.01,
) -> None:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for retrieval_two_tower. Install with: uv sync --extra deep"
        ) from exc

    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(f"{s.mlflow_experiment_prefix}-retrieval")

    df = pd.read_csv(input_path)
    if "item_id" not in df.columns:
        df = df.rename(columns={"video_id": "item_id"})
    if "is_click" in df.columns:
        df = df[df["is_click"].fillna(0).astype(int) > 0].copy()
    if "long_view" in df.columns:
        df = df[(df["long_view"].fillna(0).astype(int) > 0) | (df["is_click"] > 0)].copy()
    df = df[["user_id", "item_id"]].drop_duplicates()
    if df.empty:
        raise ValueError("no positive interactions found for two-tower training")

    interactions, user2idx, item2idx = _build_interactions(df)
    n_users = len(user2idx)
    n_items = len(item2idx)

    user_emb = torch.nn.Embedding(n_users, factors)
    item_emb = torch.nn.Embedding(n_items, factors)
    torch.nn.init.normal_(user_emb.weight, std=0.01)
    torch.nn.init.normal_(item_emb.weight, std=0.01)
    optimizer = torch.optim.Adam(list(user_emb.parameters()) + list(item_emb.parameters()), lr=lr)

    item_pool = list(range(n_items))
    rng = random.Random(42)

    with mlflow.start_run(run_name="two_tower"):
        mlflow.log_param("factors", factors)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("rows", int(len(df)))

        for _ in range(epochs):
            rng.shuffle(interactions)
            for start in range(0, len(interactions), batch_size):
                batch = interactions[start : start + batch_size]
                users = torch.tensor([x[0] for x in batch], dtype=torch.long)
                pos_items = torch.tensor([x[1] for x in batch], dtype=torch.long)
                neg_items = torch.tensor([rng.choice(item_pool) for _ in batch], dtype=torch.long)

                u = user_emb(users)
                p = item_emb(pos_items)
                n = item_emb(neg_items)
                pos_score = (u * p).sum(dim=1)
                neg_score = (u * n).sum(dim=1)
                loss = -F.logsigmoid(pos_score - neg_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        user_factors = user_emb.weight.detach().cpu().numpy().astype(np.float32)
        item_factors = item_emb.weight.detach().cpu().numpy().astype(np.float32)

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "item_factors.npy", item_factors)
        np.save(outdir / "user_factors.npy", user_factors)

        idx2item = {v: k for k, v in item2idx.items()}
        with open(outdir / "item_index.json", "w", encoding="utf-8") as f:
            json.dump({str(i): int(idx2item[i]) for i in range(len(idx2item))}, f, indent=2)
        with open(outdir / "user_index.json", "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in user2idx.items()}, f, indent=2)

        metrics = {"num_users": n_users, "num_items": n_items, "epochs": epochs}
        with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_dir=args.output_dir,
        factors=args.factors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
