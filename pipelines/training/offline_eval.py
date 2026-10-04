"""Offline evaluation of retrieval and ranking on KuaiRand-Pure.

Train on the organic log (8–21 Apr 2022). Evaluate on the later random-policy
log, which is the unbiased slice published with KuaiRand. Retrieval is
full-catalog Recall@K and NDCG@K. The ranker is pointwise AUC and log loss.
Both stages are compared with a popularity baseline fit on the same train window.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from pipelines.evaluation.metrics import ndcg_at_k, recall_at_k

LOG_COLUMNS = ["user_id", "video_id", "is_click"]
RETRIEVAL_KS = (10, 20, 50)
# Monotone: higher retrieval score, popularity, or historical CTR cannot lower p(click).
RANKER_FEATURES = ["als_score", "pop_score", "user_ctr"]
POPULARITY_FEATURES = ["pop_score"]
TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
TEST_LOG = "log_random_4_22_to_5_08_pure.csv"


def _read_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=LOG_COLUMNS,
        dtype={"user_id": np.int32, "video_id": np.int32, "is_click": np.int8},
    )
    if frame.empty:
        raise ValueError(f"{path} has no rows")
    return frame


def _positive_sets(frame: pd.DataFrame) -> dict[int, set[int]]:
    clicked = frame.loc[frame["is_click"] == 1, ["user_id", "video_id"]]
    if clicked.empty:
        return {}
    grouped = clicked.drop_duplicates().groupby("user_id")["video_id"]
    return {int(user_id): set(map(int, videos)) for user_id, videos in grouped}


def _fit_als(
    train: pd.DataFrame,
    factors: int,
    iterations: int,
    alpha: float,
    seed: int,
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
    clicked = train.loc[train["is_click"] == 1, ["user_id", "video_id"]].drop_duplicates()
    if clicked.empty:
        raise ValueError("train log has no clicks to fit ALS")
    users = np.sort(clicked["user_id"].unique())
    items = np.sort(clicked["video_id"].unique())
    user_index = {int(user_id): i for i, user_id in enumerate(users)}
    item_index = {int(item_id): i for i, item_id in enumerate(items)}
    rows = clicked["user_id"].map(user_index).to_numpy()
    cols = clicked["video_id"].map(item_index).to_numpy()
    matrix = csr_matrix(
        (np.ones(len(clicked), dtype=np.float32), (rows, cols)),
        shape=(len(users), len(items)),
    )
    threadpool_limits(limits=1, user_api="blas")
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=0.05,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
        num_threads=0,
    )
    model.fit(matrix, show_progress=False)
    return user_index, item_index, np.asarray(model.user_factors), np.asarray(model.item_factors)


def _item_popularity(train: pd.DataFrame) -> pd.Series:
    clicked = train.loc[train["is_click"] == 1, ["user_id", "video_id"]].drop_duplicates()
    if clicked.empty:
        return pd.Series(dtype=np.float64)
    return clicked.groupby("video_id")["user_id"].nunique().astype(np.float64)


def _aligned_factors(
    raw_index: dict[int, int],
    factors: np.ndarray,
    catalog: np.ndarray,
    width: int,
) -> np.ndarray:
    aligned = np.zeros((len(catalog), width), dtype=np.float32)
    catalog_index = {int(item_id): i for i, item_id in enumerate(catalog)}
    for raw_id, row in raw_index.items():
        dest = catalog_index.get(raw_id)
        if dest is not None:
            aligned[dest] = factors[row]
    return aligned


def _retrieval_sums_for_scores(
    user_ids: list[int],
    scores: np.ndarray,
    catalog: np.ndarray,
    relevant: dict[int, set[int]],
    masked: dict[int, set[int]],
    ks: tuple[int, ...],
) -> tuple[dict[int, float], dict[int, float], int]:
    recall_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    counted = 0
    k_max = max(ks)
    catalog_index = {int(item_id): i for i, item_id in enumerate(catalog)}
    for row, user_id in enumerate(user_ids):
        row_scores = scores[row].copy()
        for item_id in masked.get(user_id, ()):
            col = catalog_index.get(item_id)
            if col is not None:
                row_scores[col] = -np.inf
        if not np.isfinite(row_scores).any():
            continue
        k_take = min(k_max, int(row_scores.size))
        top_local = np.argpartition(row_scores, -k_take)[-k_take:]
        top_local = top_local[np.argsort(row_scores[top_local])[::-1]]
        ranked = catalog[top_local]
        rel = relevant[user_id]
        counted += 1
        for k in ks:
            recall_sums[k] += recall_at_k(ranked, rel, k)
            ndcg_sums[k] += ndcg_at_k(ranked, rel, k)
    return recall_sums, ndcg_sums, counted


def evaluate_retrieval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity: pd.Series,
    ks: tuple[int, ...] = RETRIEVAL_KS,
    batch_size: int = 2048,
) -> dict[str, object]:
    train_pos = _positive_sets(train)
    test_pos = _positive_sets(test)
    catalog_ids = sorted(set(train["video_id"].tolist()) | set(test["video_id"].tolist()))
    catalog = np.asarray(catalog_ids, dtype=np.int32)
    factors = int(user_factors.shape[1])
    als_items = _aligned_factors(item_index, item_factors, catalog, factors)
    pop_scores = np.zeros(len(catalog), dtype=np.float64)
    catalog_index = {int(item_id): i for i, item_id in enumerate(catalog)}
    for item_id, score in popularity.items():
        col = catalog_index.get(int(item_id))
        if col is not None:
            pop_scores[col] = float(score)
    # Stable tie-break toward smaller item ids.
    pop_scores = pop_scores + (1.0 - np.arange(len(catalog)) / max(len(catalog), 1)) * 1e-9

    eval_users: list[int] = []
    relevant: dict[int, set[int]] = {}
    for user_id, items in test_pos.items():
        if user_id not in user_index:
            continue
        held_out = items - train_pos.get(user_id, set())
        if not held_out:
            continue
        eval_users.append(user_id)
        relevant[user_id] = held_out
    eval_users.sort()

    als_recall_sums = {k: 0.0 for k in ks}
    als_ndcg_sums = {k: 0.0 for k in ks}
    pop_recall_sums = {k: 0.0 for k in ks}
    pop_ndcg_sums = {k: 0.0 for k in ks}
    counted_users = 0

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start : start + batch_size]
        user_rows = [user_index[user_id] for user_id in batch_users]
        als_scores = user_factors[user_rows] @ als_items.T
        pop_batch = np.broadcast_to(pop_scores, (len(batch_users), len(catalog))).copy()
        als_recall, als_ndcg, als_n = _retrieval_sums_for_scores(
            batch_users, als_scores, catalog, relevant, train_pos, ks
        )
        pop_recall, pop_ndcg, pop_n = _retrieval_sums_for_scores(
            batch_users, pop_batch, catalog, relevant, train_pos, ks
        )
        if als_n != pop_n:
            raise RuntimeError("ALS and popularity scored different user batches")
        counted_users += als_n
        for k in ks:
            als_recall_sums[k] += als_recall[k]
            als_ndcg_sums[k] += als_ndcg[k]
            pop_recall_sums[k] += pop_recall[k]
            pop_ndcg_sums[k] += pop_ndcg[k]

    als_metrics: dict[str, float] = {}
    pop_metrics: dict[str, float] = {}
    denom = float(counted_users) if counted_users else 1.0
    for k in ks:
        als_metrics[f"recall@{k}"] = round(als_recall_sums[k] / denom, 6)
        als_metrics[f"ndcg@{k}"] = round(als_ndcg_sums[k] / denom, 6)
        pop_metrics[f"recall@{k}"] = round(pop_recall_sums[k] / denom, 6)
        pop_metrics[f"ndcg@{k}"] = round(pop_ndcg_sums[k] / denom, 6)

    return {
        "protocol": (
            "Full catalog. Warm users only (user had a train click). "
            "Relevance is test clicks the user did not already click in train. "
            "Already-clicked train items are removed from the ranked list. "
            "Macro-average over users."
        ),
        "k": list(ks),
        "n_users": counted_users,
        "n_items": int(len(catalog)),
        "models": {"als": als_metrics, "popularity": pop_metrics},
    }


def _smoothed_rate(clicks: np.ndarray, impressions: np.ndarray) -> np.ndarray:
    return (clicks + 1.0) / (impressions + 2.0)


def _ranker_frame(
    frame: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity: pd.Series,
    user_clicks: pd.Series,
    user_impr: pd.Series,
    global_ctr: float,
) -> pd.DataFrame:
    out = pd.DataFrame({"is_click": frame["is_click"].astype(np.int8)})
    video_ids = frame["video_id"].to_numpy()
    user_ids = frame["user_id"].to_numpy()
    pop = popularity.reindex(video_ids).fillna(0.0).to_numpy(dtype=np.float64)
    out["pop_score"] = np.log1p(pop).astype(np.float32)
    u_clicks = user_clicks.reindex(user_ids).fillna(0.0).to_numpy(dtype=np.float64)
    u_impr = user_impr.reindex(user_ids).fillna(0.0).to_numpy(dtype=np.float64)
    known_user = user_impr.reindex(user_ids).notna().to_numpy()
    user_ctr = np.full(len(frame), global_ctr, dtype=np.float64)
    user_ctr[known_user] = _smoothed_rate(u_clicks[known_user], u_impr[known_user])
    out["user_ctr"] = user_ctr.astype(np.float32)

    als = np.zeros(len(frame), dtype=np.float32)
    user_codes = pd.Series(user_ids).map(user_index)
    item_codes = pd.Series(video_ids).map(item_index)
    known_pairs = (user_codes.notna() & item_codes.notna()).to_numpy()
    if known_pairs.any():
        u_rows = user_codes[known_pairs].to_numpy(dtype=np.int32)
        i_rows = item_codes[known_pairs].to_numpy(dtype=np.int32)
        als[known_pairs] = np.sum(
            user_factors[u_rows] * item_factors[i_rows], axis=1, dtype=np.float32
        )
    out["als_score"] = als
    return out


def _fit_classifier(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    features: list[str],
    iterations: int,
    depth: int,
    learning_rate: float,
    seed: int,
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=5.0,
        monotone_constraints=[1] * len(features),
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_x[features], train_y)
    return model


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        raise ValueError("ranker evaluation needs both click classes")
    return {
        "auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "logloss": round(float(log_loss(y_true, probabilities, labels=[0, 1])), 6),
    }


def evaluate_ranker(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity: pd.Series,
    iterations: int,
    depth: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    user_clicks = train.groupby("user_id")["is_click"].sum()
    user_impr = train.groupby("user_id")["is_click"].size()
    global_ctr = float(train["is_click"].mean())
    shared = dict(
        user_index=user_index,
        item_index=item_index,
        user_factors=user_factors,
        item_factors=item_factors,
        popularity=popularity,
        user_clicks=user_clicks,
        user_impr=user_impr,
        global_ctr=global_ctr,
    )
    print("building ranker frames", flush=True)
    train_x = _ranker_frame(train, **shared)
    test_x = _ranker_frame(test, **shared)
    y_train = train_x["is_click"].to_numpy()
    y_test = test_x["is_click"].to_numpy()
    print("fitting catboost ranker", flush=True)
    ranker = _fit_classifier(
        train_x, y_train, RANKER_FEATURES, iterations, depth, learning_rate, seed
    )
    print("fitting popularity baseline", flush=True)
    baseline = _fit_classifier(
        train_x, y_train, POPULARITY_FEATURES, iterations, depth, learning_rate, seed
    )
    ranker_prob = ranker.predict_proba(test_x[RANKER_FEATURES])[:, 1]
    baseline_prob = baseline.predict_proba(test_x[POPULARITY_FEATURES])[:, 1]
    return {
        "protocol": (
            "Pointwise click model. Fit on every organic-log impression. "
            "Scored on every random-policy impression. "
            "ALS factors, item popularity, and user CTR use the train window only. "
            "Scores are constrained to be non-decreasing in each feature. "
            "The popularity baseline is the same model fit on log item popularity only."
        ),
        "features": {
            "catboost": RANKER_FEATURES,
            "popularity": POPULARITY_FEATURES,
        },
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "train_click_rate": round(float(y_train.mean()), 6),
        "test_click_rate": round(float(y_test.mean()), 6),
        "models": {
            "catboost": _classification_metrics(y_test, ranker_prob),
            "popularity": _classification_metrics(y_test, baseline_prob),
        },
    }


def run_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factors: int,
    als_iterations: int,
    als_alpha: float,
    ranker_iterations: int,
    ranker_depth: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    print(
        f"train_rows={len(train)} test_rows={len(test)} "
        f"train_users={train['user_id'].nunique()} test_users={test['user_id'].nunique()}",
        flush=True,
    )
    print("fitting ALS", flush=True)
    user_index, item_index, user_factors, item_factors = _fit_als(
        train, factors, als_iterations, als_alpha, seed
    )
    popularity = _item_popularity(train)
    print("scoring retrieval", flush=True)
    retrieval = evaluate_retrieval(
        train,
        test,
        user_index,
        item_index,
        user_factors,
        item_factors,
        popularity,
    )
    print(json.dumps(retrieval["models"]), flush=True)
    ranker = evaluate_ranker(
        train,
        test,
        user_index,
        item_index,
        user_factors,
        item_factors,
        popularity,
        ranker_iterations,
        ranker_depth,
        learning_rate,
        seed,
    )
    print(json.dumps(ranker["models"]), flush=True)
    import catboost
    import implicit
    import sklearn

    return {
        "dataset": "KuaiRand-Pure",
        "source": "https://zenodo.org/records/10439422",
        "train_log": TRAIN_LOG,
        "test_log": TEST_LOG,
        "label": "is_click",
        "config": {
            "factors": factors,
            "als_iterations": als_iterations,
            "als_regularization": 0.05,
            "als_alpha": als_alpha,
            "ranker_iterations": ranker_iterations,
            "ranker_depth": ranker_depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": 5.0,
            "monotone_constraints": True,
            "seed": seed,
        },
        "retrieval": retrieval,
        "ranker": ranker,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "implicit": implicit.__version__,
            "catboost": catboost.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def _synthetic_logs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | float]] = []
    for user_id in range(40):
        liked = set(rng.choice(20, size=4, replace=False).tolist())
        for _ in range(25):
            video_id = int(rng.integers(0, 20))
            prefer = video_id in liked or video_id < 3
            click = int(rng.random() < (0.7 if prefer else 0.1))
            rows.append(
                {
                    "user_id": user_id,
                    "video_id": video_id,
                    "is_click": click,
                }
            )
    train = pd.DataFrame(rows)
    test_rows: list[dict[str, int | float]] = []
    for user_id in range(40):
        liked = {user_id % 20, (user_id + 1) % 20}
        for _ in range(12):
            video_id = int(rng.integers(0, 20))
            click = int(rng.random() < (0.65 if video_id in liked or video_id < 2 else 0.15))
            test_rows.append(
                {
                    "user_id": user_id,
                    "video_id": video_id,
                    "is_click": click,
                }
            )
    return train, pd.DataFrame(test_rows)


def _load_kuairand(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / TRAIN_LOG
    test_path = data_dir / TEST_LOG
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path.name} and {test_path.name} under {data_dir}. "
            "Run bash scripts/download_kuairand_pure.sh"
        )
    print(f"reading {train_path}", flush=True)
    train = _read_log(train_path)
    print(f"reading {test_path}", flush=True)
    test = _read_log(test_path)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/KuaiRand-Pure/data"))
    parser.add_argument("--output", type=Path, default=Path("results/kuairand_pure_metrics.json"))
    parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic end-to-end eval")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--als-iterations", type=int, default=15)
    parser.add_argument("--als-alpha", type=float, default=40.0)
    parser.add_argument("--ranker-iterations", type=int, default=150)
    parser.add_argument("--ranker-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        train, test = _synthetic_logs(args.seed)
        factors = min(args.factors, 8)
        als_iterations = min(args.als_iterations, 3)
        ranker_iterations = min(args.ranker_iterations, 10)
        ranker_depth = min(args.ranker_depth, 3)
    else:
        train, test = _load_kuairand(args.data_dir)
        factors = args.factors
        als_iterations = args.als_iterations
        ranker_iterations = args.ranker_iterations
        ranker_depth = args.ranker_depth

    payload = run_eval(
        train,
        test,
        factors=factors,
        als_iterations=als_iterations,
        als_alpha=args.als_alpha,
        ranker_iterations=ranker_iterations,
        ranker_depth=ranker_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    if args.smoke:
        payload["dataset"] = "synthetic-smoke"
        payload["train_log"] = "synthetic"
        payload["test_log"] = "synthetic"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
