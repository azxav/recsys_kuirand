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
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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


def _read_log(path: Path, extra: bool = False) -> pd.DataFrame:
    columns = ["user_id", "video_id", "is_click"]
    dtype: dict[str, type] = {"user_id": np.int32, "video_id": np.int32, "is_click": np.int8}
    if extra:
        columns.extend(["date", "play_time_ms"])
        dtype["date"] = np.int32
        dtype["play_time_ms"] = np.float32
    frame = pd.read_csv(path, usecols=columns, dtype=dtype)
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
    retrieval_scores: np.ndarray | None = None,
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
    if retrieval_scores is not None:
        out["als_score"] = np.asarray(retrieval_scores, dtype=np.float32)
        return out

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


def _clip_prob(probabilities: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.clip(probabilities, 1e-6, 1.0 - 1e-6))


def _ranker_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    retrieval_model: Any,
    iterations: int,
    depth: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    popularity = _item_popularity(train_df)
    user_clicks = train_df.groupby("user_id")["is_click"].sum()
    user_impr = train_df.groupby("user_id")["is_click"].size()
    global_ctr = float(train_df["is_click"].mean())
    train_scores = retrieval_model.pair_scores(
        train_df["user_id"].to_numpy(), train_df["video_id"].to_numpy()
    )
    test_scores = retrieval_model.pair_scores(
        test_df["user_id"].to_numpy(), test_df["video_id"].to_numpy()
    )
    empty = np.zeros((1, 1), dtype=np.float32)
    train_x = _ranker_frame(
        train_df,
        retrieval_model.user_index,
        retrieval_model.item_index,
        empty,
        empty,
        popularity,
        user_clicks,
        user_impr,
        global_ctr,
        retrieval_scores=train_scores,
    )
    test_x = _ranker_frame(
        test_df,
        retrieval_model.user_index,
        retrieval_model.item_index,
        empty,
        empty,
        popularity,
        user_clicks,
        user_impr,
        global_ctr,
        retrieval_scores=test_scores,
    )
    y_train = train_x["is_click"].to_numpy()
    y_test = test_x["is_click"].to_numpy()
    ranker = _fit_classifier(
        train_x, y_train, RANKER_FEATURES, iterations, depth, learning_rate, seed
    )
    baseline = _fit_classifier(
        train_x, y_train, POPULARITY_FEATURES, iterations, depth, learning_rate, seed
    )
    return (
        y_test,
        ranker.predict_proba(test_x[RANKER_FEATURES])[:, 1],
        baseline.predict_proba(test_x[POPULARITY_FEATURES])[:, 1],
    )


def _calibrated_report(
    y_val: np.ndarray,
    raw_val: np.ndarray,
    y_test: np.ndarray,
    raw_test: np.ndarray,
) -> dict[str, object]:
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_val, y_val)
    platt = LogisticRegression(max_iter=500)
    platt.fit(raw_val.reshape(-1, 1), y_val)
    calibrated = {
        "isotonic": isotonic.predict(raw_test),
        "platt": platt.predict_proba(raw_test.reshape(-1, 1))[:, 1],
    }
    val_probs = {
        "isotonic": isotonic.predict(raw_val),
        "platt": platt.predict_proba(raw_val.reshape(-1, 1))[:, 1],
    }
    val_losses = {
        name: round(float(log_loss(y_val, _clip_prob(prob), labels=[0, 1])), 6)
        for name, prob in val_probs.items()
    }
    method = min(val_losses, key=lambda name: val_losses[name])
    return {
        "method": method,
        "selected_by": "validation log loss",
        "validation_logloss": val_losses,
        "before": _classification_metrics(y_test, _clip_prob(raw_test)),
        "after": _classification_metrics(
            y_test, _clip_prob(np.asarray(calibrated[method], dtype=float))
        ),
    }


def _fit_and_calibrate(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    als_model: Any,
    selected_model: Any,
    iterations: int,
    depth: int,
    learning_rate: float,
    seed: int,
    validation_start: int,
) -> dict[str, object]:
    """Fit on the pre-validation organic slice and calibrate on the held-out organic days."""

    def predict_both(model: Any) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Two calls refit CatBoost. Seed makes that deterministic, but it is slower
        # and the val/test models would match. Fit once by scoring both frames after one fit.
        popularity = _item_popularity(fit_df)
        user_clicks = fit_df.groupby("user_id")["is_click"].sum()
        user_impr = fit_df.groupby("user_id")["is_click"].size()
        global_ctr = float(fit_df["is_click"].mean())
        empty = np.zeros((1, 1), dtype=np.float32)
        frames = {}
        for name, frame in (("train", fit_df), ("val", val_df), ("test", test_df)):
            scores = model.pair_scores(frame["user_id"].to_numpy(), frame["video_id"].to_numpy())
            built = _ranker_frame(
                frame,
                model.user_index,
                model.item_index,
                empty,
                empty,
                popularity,
                user_clicks,
                user_impr,
                global_ctr,
                retrieval_scores=scores,
            )
            frames[name] = built
        y_train = frames["train"]["is_click"].to_numpy()
        ranker = _fit_classifier(
            frames["train"], y_train, RANKER_FEATURES, iterations, depth, learning_rate, seed
        )
        baseline = _fit_classifier(
            frames["train"], y_train, POPULARITY_FEATURES, iterations, depth, learning_rate, seed
        )
        packed = {}
        for name in ("val", "test"):
            packed[name] = (
                frames[name]["is_click"].to_numpy(),
                ranker.predict_proba(frames[name][RANKER_FEATURES])[:, 1],
                baseline.predict_proba(frames[name][POPULARITY_FEATURES])[:, 1],
            )
        return packed

    als_pred = predict_both(als_model)
    selected_pred = predict_both(selected_model)
    y_val, als_val, pop_val = als_pred["val"]
    y_test, als_test, pop_test = als_pred["test"]
    _, selected_val, _ = selected_pred["val"]
    y_selected_test, selected_test, _ = selected_pred["test"]
    if not np.array_equal(y_test, y_selected_test):
        raise RuntimeError("calibration targets changed between retrievers")
    return {
        "train_dates": f"< {validation_start}",
        "calibration_dates": f">= {validation_start}",
        "note": (
            "Models are fit only on organic rows before the validation date. "
            "Isotonic and Platt are fit on the later organic rows. "
            "The method with lower validation log loss is applied to the random-policy test. "
            "Test labels are not used to choose the calibrator."
        ),
        "catboost_als_previous": _calibrated_report(y_val, als_val, y_test, als_test),
        "catboost_selected_retriever": _calibrated_report(
            y_val, selected_val, y_test, selected_test
        ),
        "popularity": _calibrated_report(y_val, pop_val, y_test, pop_test),
    }


def run_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selection: dict[str, object],
    ranker_iterations: int,
    ranker_depth: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    from pipelines.training.retrieval_tune import (
        VAL_START_DATE,
        _blend_scores,
        build_eval_context,
        metrics_for_scores,
    )
    from pipelines.training.retrievers import OLD_ALS, fit_als, fit_ease, fit_itemknn

    print(
        f"train_rows={len(train)} test_rows={len(test)} "
        f"train_users={train['user_id'].nunique()} test_users={test['user_id'].nunique()}",
        flush=True,
    )
    chosen = selection["chosen"]
    assert isinstance(chosen, dict)
    als_cfg = chosen["als_tuned"]["config"]
    knn_cfg = chosen["itemknn"]["config"]
    ease_cfg = chosen["ease"]["config"]
    hybrid_cfg = chosen["hybrid"]["config"]
    assert isinstance(als_cfg, dict)
    assert isinstance(knn_cfg, dict)
    assert isinstance(ease_cfg, dict)
    assert isinstance(hybrid_cfg, dict)

    print("refitting retrievers on the full organic log", flush=True)
    als_previous = fit_als(
        train,
        factors=int(OLD_ALS["factors"]),
        regularization=float(OLD_ALS["regularization"]),
        iterations=int(OLD_ALS["iterations"]),
        alpha=float(OLD_ALS["alpha"]),
        weighting=str(OLD_ALS["weighting"]),
        seed=seed,
    )
    als_tuned = fit_als(
        train,
        factors=int(als_cfg["factors"]),
        regularization=float(als_cfg["regularization"]),
        iterations=int(als_cfg["iterations"]),
        alpha=float(als_cfg["alpha"]),
        weighting=str(als_cfg["weighting"]),
        seed=seed,
    )
    itemknn = fit_itemknn(train, kind=str(knn_cfg["kind"]), neighbors=int(knn_cfg["neighbors"]))
    ease = fit_ease(
        train, reg_lambda=float(ease_cfg["lambda"]), weighting=str(ease_cfg["weighting"])
    )
    fitted: dict[str, Any] = {"als_tuned": als_tuned, "itemknn": itemknn, "ease": ease}
    ctx = build_eval_context(train, test)
    hybrid_base = fitted[str(hybrid_cfg["base"])]
    hybrid_weight = float(hybrid_cfg["weight"])
    hybrid_kind = str(hybrid_cfg["blend"])

    def hybrid_batch(batch: list[int]) -> np.ndarray:
        collaborative = hybrid_base.score_batch(batch, ctx.catalog)
        return _blend_scores(collaborative, ctx.pop_scores, hybrid_weight, hybrid_kind)

    scorers = {
        "popularity": lambda batch: np.broadcast_to(
            ctx.pop_scores, (len(batch), len(ctx.catalog))
        ).copy(),
        "als": lambda batch: als_previous.score_batch(batch, ctx.catalog),
        "als_tuned": lambda batch: als_tuned.score_batch(batch, ctx.catalog),
        "itemknn": lambda batch: itemknn.score_batch(batch, ctx.catalog),
        "ease": lambda batch: ease.score_batch(batch, ctx.catalog),
        "hybrid": hybrid_batch,
    }
    print("scoring retrieval on the random-policy log", flush=True)
    model_metrics = {name: metrics_for_scores(scorer, ctx) for name, scorer in scorers.items()}
    print(json.dumps(model_metrics), flush=True)
    popularity = _item_popularity(train)
    previous = evaluate_retrieval(
        train,
        test,
        als_previous.user_index,
        als_previous.item_index,
        als_previous.user_factors,
        als_previous.item_factors,
        popularity,
    )
    previous_models = previous["models"]
    assert isinstance(previous_models, dict)
    for model_name in ("als", "popularity"):
        published = previous_models[model_name]
        assert isinstance(published, dict)
        for key, value in published.items():
            if model_metrics[model_name][key] != value:
                raise RuntimeError(
                    f"retrieval metric drift for {model_name} {key}: "
                    f"{model_metrics[model_name][key]} vs {value}"
                )
    retrieval = {
        "protocol": (
            "Full catalog. Warm users only (user had an organic click). "
            "Relevance is random-policy clicks the user did not already click in the organic log. "
            "Already-clicked organic items are removed from the ranked list. "
            "Macro-average over users. "
            "als is the previously published configuration. "
            "als_tuned, itemknn, ease, and hybrid were selected on the organic validation "
            f"slice (dates >= {VAL_START_DATE}) and refit on the full organic log."
        ),
        "k": list(RETRIEVAL_KS),
        "n_users": int(cast(int, previous["n_users"])),
        "n_items": int(cast(int, previous["n_items"])),
        "models": model_metrics,
    }

    print("fitting uncalibrated ranker on the full organic log", flush=True)
    y_test, ranker_prob, baseline_prob = _ranker_on_split(
        train, test, als_previous, ranker_iterations, ranker_depth, learning_rate, seed
    )
    ranker_full = {
        "protocol": (
            "Pointwise click model fit on every organic impression and scored on every "
            "random-policy impression. Retrieval feature is the previous ALS dot product. "
            "No probability calibration."
        ),
        "features": {"catboost": RANKER_FEATURES, "popularity": POPULARITY_FEATURES},
        "retrieval_feature": "als_previous",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_click_rate": round(float(train["is_click"].mean()), 6),
        "test_click_rate": round(float(test["is_click"].mean()), 6),
        "models": {
            "catboost": _classification_metrics(y_test, ranker_prob),
            "popularity": _classification_metrics(y_test, baseline_prob),
        },
    }
    print(json.dumps(ranker_full["models"]), flush=True)

    retriever_name = str(selection["ranker_retriever"])
    print(f"refitting ranker with {retriever_name} scores", flush=True)
    y_test_new, ranker_prob_new, baseline_prob_new = _ranker_on_split(
        train,
        test,
        fitted[retriever_name],
        ranker_iterations,
        ranker_depth,
        learning_rate,
        seed,
    )
    ranker_improved = {
        "protocol": (
            "Same full-organic training protocol as the uncalibrated ranker. "
            f"The retrieval feature is {retriever_name}, the pure retriever with the "
            "best organic validation objective."
        ),
        "retrieval_feature": retriever_name,
        "models": {
            "catboost": _classification_metrics(y_test_new, ranker_prob_new),
            "popularity": _classification_metrics(y_test_new, baseline_prob_new),
        },
    }
    print(json.dumps(ranker_improved["models"]), flush=True)

    fit_df = train.loc[train["date"] < VAL_START_DATE]
    val_df = train.loc[train["date"] >= VAL_START_DATE]
    print("calibrating rankers on the organic validation slice", flush=True)
    als_fit = fit_als(
        fit_df,
        factors=int(OLD_ALS["factors"]),
        regularization=float(OLD_ALS["regularization"]),
        iterations=int(OLD_ALS["iterations"]),
        alpha=float(OLD_ALS["alpha"]),
        weighting=str(OLD_ALS["weighting"]),
        seed=seed,
    )
    selected_fit = {
        "als_tuned": lambda frame: fit_als(
            frame,
            factors=int(als_cfg["factors"]),
            regularization=float(als_cfg["regularization"]),
            iterations=int(als_cfg["iterations"]),
            alpha=float(als_cfg["alpha"]),
            weighting=str(als_cfg["weighting"]),
            seed=seed,
        ),
        "itemknn": lambda frame: fit_itemknn(
            frame, kind=str(knn_cfg["kind"]), neighbors=int(knn_cfg["neighbors"])
        ),
        "ease": lambda frame: fit_ease(
            frame, reg_lambda=float(ease_cfg["lambda"]), weighting=str(ease_cfg["weighting"])
        ),
    }[retriever_name](fit_df)
    calibration = _fit_and_calibrate(
        fit_df,
        val_df,
        test,
        als_fit,
        selected_fit,
        ranker_iterations,
        ranker_depth,
        learning_rate,
        seed,
        VAL_START_DATE,
    )
    print(json.dumps(calibration), flush=True)

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
            "seed": seed,
            "als_previous": OLD_ALS,
            "ranker_iterations": ranker_iterations,
            "ranker_depth": ranker_depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": 5.0,
            "monotone_constraints": True,
        },
        "retrieval_selection": selection,
        "retrieval": retrieval,
        "ranker": {
            "uncalibrated_full_train": ranker_full,
            "with_selected_retriever": ranker_improved,
            "calibration": calibration,
        },
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
                    "date": 20220410 if len(rows) % 25 < 18 else 20220420,
                    "play_time_ms": float(rng.integers(1000, 20000)),
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
    train = _read_log(train_path, extra=True)
    print(f"reading {test_path}", flush=True)
    test = _read_log(test_path)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/KuaiRand-Pure/data"))
    parser.add_argument("--output", type=Path, default=Path("results/kuairand_pure_metrics.json"))
    parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic end-to-end eval")
    parser.add_argument("--ranker-iterations", type=int, default=150)
    parser.add_argument("--ranker-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from pipelines.training.retrieval_tune import tune_retrievers

    if args.smoke:
        train, test = _synthetic_logs(args.seed)
        ranker_iterations = min(args.ranker_iterations, 10)
        ranker_depth = min(args.ranker_depth, 3)
        smoke = True
    else:
        train_path = args.data_dir / TRAIN_LOG
        print(f"reading {train_path}", flush=True)
        train = _read_log(train_path, extra=True)
        test = None
        ranker_iterations = args.ranker_iterations
        ranker_depth = args.ranker_depth
        smoke = False
    # Hyperparameters are chosen before the random-policy log is read.
    selection = tune_retrievers(train, seed=args.seed, smoke=smoke)
    if test is None:
        test_path = args.data_dir / TEST_LOG
        print(f"reading {test_path}", flush=True)
        test = _read_log(test_path)
    payload = run_eval(
        train,
        test,
        selection,
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
