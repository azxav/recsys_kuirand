import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[int, int]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def _topk(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]


def _score_model(
    user_id: int,
    user_index: dict[int, int],
    item_index: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int,
) -> dict[int, float]:
    uidx = user_index.get(user_id)
    if uidx is None:
        return {}
    scores = item_factors @ user_factors[uidx]
    return {
        item_index[i]: s
        for i, s in _topk(scores, k)
        if i in item_index
    }


def _load_factor_bundle(
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    user = np.load(base_dir / "user_factors.npy")
    item = np.load(base_dir / "item_factors.npy")
    user_idx = _load_json(base_dir / "user_index.json")
    item_idx = _load_json(base_dir / "item_index.json")
    return user, item, user_idx, item_idx


def run(
    interactions_path: str,
    als_dir: str,
    bpr_dir: str,
    output_path: str,
    two_tower_dir: str | None = None,
    candidate_k: int = 500,
) -> None:
    df = pd.read_csv(interactions_path)
    if "item_id" not in df.columns:
        df = df.rename(columns={"video_id": "item_id", "play_time_ms": "watch_time_ms"})

    als_user, als_item, als_user_idx, als_item_idx = _load_factor_bundle(Path(als_dir))
    bpr_user, bpr_item, bpr_user_idx, bpr_item_idx = _load_factor_bundle(Path(bpr_dir))

    two_user: np.ndarray | None = None
    two_item: np.ndarray | None = None
    two_user_idx: dict[int, int] = {}
    two_item_idx: dict[int, int] = {}
    if two_tower_dir is not None:
        two_dir = Path(two_tower_dir)
        if (two_dir / "user_factors.npy").exists() and (two_dir / "item_factors.npy").exists():
            two_user, two_item, two_user_idx, two_item_idx = _load_factor_bundle(two_dir)

    random_exposure_agg = ("is_rand", "max") if "is_rand" in df.columns else ("is_click", "min")
    labels = (
        df.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            is_click=("is_click", "max"),
            long_view=("long_view", "max"),
            watch_time_ms=("watch_time_ms", "max"),
            tab=("tab", "max"),
            is_random_exposure=random_exposure_agg,
        )
    )

    rows: list[dict[str, int | float | str]] = []
    for qid, user_id in enumerate(sorted(df["user_id"].unique().tolist()), start=1):
        als_scores = _score_model(
            user_id,
            als_user_idx,
            als_item_idx,
            als_user,
            als_item,
            candidate_k,
        )
        bpr_scores = _score_model(
            user_id,
            bpr_user_idx,
            bpr_item_idx,
            bpr_user,
            bpr_item,
            candidate_k,
        )

        two_scores: dict[int, float] = {}
        if two_user is not None and two_item is not None:
            two_scores = _score_model(
                user_id,
                two_user_idx,
                two_item_idx,
                two_user,
                two_item,
                candidate_k,
            )

        union_items = set(als_scores) | set(bpr_scores) | set(two_scores)
        user_labels = labels[labels["user_id"] == user_id].set_index("item_id")

        for item_id in union_items:
            if item_id in user_labels.index:
                row = user_labels.loc[item_id]
                is_click = int(row["is_click"])
                long_view = int(row["long_view"])
                watch_time_ms = int(row["watch_time_ms"])
                tab = int(row["tab"])
                is_random_exposure = int(row["is_random_exposure"])
            else:
                is_click = 0
                long_view = 0
                watch_time_ms = 0
                tab = 0
                is_random_exposure = 0

            rows.append(
                {
                    "query_id": f"{user_id}-{qid}",
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "als_score": float(als_scores.get(item_id, 0.0)),
                    "bpr_score": float(bpr_scores.get(item_id, 0.0)),
                    "two_tower_score": float(two_scores.get(item_id, 0.0)),
                    "tab": tab,
                    "is_click": is_click,
                    "long_view": long_view,
                    "watch_time_ms": watch_time_ms,
                    "is_random_exposure": is_random_exposure,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--als-dir", required=True)
    parser.add_argument("--bpr-dir", required=True)
    parser.add_argument("--two-tower-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-k", type=int, default=500)
    args = parser.parse_args()
    run(
        interactions_path=args.interactions,
        als_dir=args.als_dir,
        bpr_dir=args.bpr_dir,
        two_tower_dir=args.two_tower_dir,
        output_path=args.output,
        candidate_k=args.candidate_k,
    )
