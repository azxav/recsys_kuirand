import argparse

import numpy as np
import pandas as pd


def hit_rate_at_k(df: pd.DataFrame, k: int = 10) -> float:
    hits = 0
    total = 0
    for _, grp in df.groupby("query_id"):
        total += 1
        top = grp.sort_values("final_score", ascending=False).head(k)
        if (top["is_click"] > 0).any() or (top["long_view"] > 0).any():
            hits += 1
    return hits / max(total, 1)


def mrr_at_k(df: pd.DataFrame, k: int = 10) -> float:
    vals: list[float] = []
    for _, grp in df.groupby("query_id"):
        top = grp.sort_values("final_score", ascending=False).head(k).reset_index(drop=True)
        rel = (top["is_click"] > 0) | (top["long_view"] > 0)
        idx = np.where(rel.to_numpy())[0]
        vals.append(1.0 / float(idx[0] + 1) if len(idx) > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(df: pd.DataFrame, k: int = 10) -> float:
    vals: list[float] = []
    for _, grp in df.groupby("query_id"):
        g = grp.copy()
        g["gain"] = g["long_view"].astype(float) + 0.5 * g["is_click"].astype(float)
        top = g.sort_values("final_score", ascending=False).head(k)
        ideal = g.sort_values("gain", ascending=False).head(k)

        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum(top["gain"].to_numpy() * discounts))
        idcg = float(np.sum(ideal["gain"].to_numpy() * discounts))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def run(predictions_path: str) -> None:
    df = pd.read_csv(predictions_path)
    if "final_score" not in df.columns:
        two_tower = df["two_tower_score"] if "two_tower_score" in df.columns else 0.0
        df["final_score"] = (
            0.30 * df["pred_click"]
            + 0.30 * df["pred_long_view"]
            + 0.25 * df["pred_watch_time"]
            + 0.15 * two_tower
        )

    print({
        "hit_rate@10": hit_rate_at_k(df, 10),
        "mrr@10": mrr_at_k(df, 10),
        "ndcg@10": ndcg_at_k(df, 10),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    run(predictions_path=args.predictions)
