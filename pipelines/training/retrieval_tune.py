"""Pick retrieval hyperparameters on a time split of the organic log.

The random-policy test log is not read here. The objective is fixed:
the mean of Recall@10, Recall@20, Recall@50, and NDCG@50.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from pipelines.training.offline_eval import (
    RETRIEVAL_KS,
    _item_popularity,
    _positive_sets,
    _retrieval_sums_for_scores,
)
from pipelines.training.retrievers import fit_als, fit_ease, fit_itemknn

VAL_START_DATE = 20220418
OBJECTIVE_KEYS = ("recall@10", "recall@20", "recall@50", "ndcg@50")


def selection_score(metrics: dict[str, float]) -> float:
    return float(sum(metrics[key] for key in OBJECTIVE_KEYS) / len(OBJECTIVE_KEYS))


@dataclass
class EvalContext:
    catalog: np.ndarray
    user_ids: list[int]
    relevant: dict[int, set[int]]
    masked: dict[int, set[int]]
    pop_scores: np.ndarray


def build_eval_context(train: pd.DataFrame, evaluation: pd.DataFrame) -> EvalContext:
    train_pos = _positive_sets(train)
    eval_pos = _positive_sets(evaluation)
    catalog_ids = sorted(set(train["video_id"].tolist()) | set(evaluation["video_id"].tolist()))
    catalog = np.asarray(catalog_ids, dtype=np.int32)
    catalog_index = {int(item_id): i for i, item_id in enumerate(catalog)}
    popularity = _item_popularity(train)
    pop_scores = np.zeros(len(catalog), dtype=np.float64)
    for item_id, score in popularity.items():
        col = catalog_index.get(int(item_id))
        if col is not None:
            pop_scores[col] = float(score)
    pop_scores = pop_scores + (1.0 - np.arange(len(catalog)) / max(len(catalog), 1)) * 1e-9

    user_ids: list[int] = []
    relevant: dict[int, set[int]] = {}
    for user_id, items in eval_pos.items():
        if user_id not in train_pos:
            continue
        held_out = items - train_pos[user_id]
        if not held_out:
            continue
        user_ids.append(user_id)
        relevant[user_id] = held_out
    user_ids.sort()
    return EvalContext(
        catalog=catalog,
        user_ids=user_ids,
        relevant=relevant,
        masked=train_pos,
        pop_scores=pop_scores,
    )


def metrics_for_scores(
    scores_for_batch: Callable[[list[int]], np.ndarray],
    ctx: EvalContext,
    ks: tuple[int, ...] = RETRIEVAL_KS,
    batch_size: int = 2048,
) -> dict[str, float]:
    recall_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    counted = 0
    for start in range(0, len(ctx.user_ids), batch_size):
        batch = ctx.user_ids[start : start + batch_size]
        scores = scores_for_batch(batch)
        batch_recall, batch_ndcg, batch_n = _retrieval_sums_for_scores(
            batch, scores, ctx.catalog, ctx.relevant, ctx.masked, ks
        )
        counted += batch_n
        for k in ks:
            recall_sums[k] += batch_recall[k]
            ndcg_sums[k] += batch_ndcg[k]
    denom = float(counted) if counted else 1.0
    return {
        f"{name}@{k}": round(sums[k] / denom, 6)
        for name, sums in (
            ("recall", recall_sums),
            ("ndcg", ndcg_sums),
        )
        for k in ks
    }


def als_grid(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "factors": 4,
                "regularization": 0.05,
                "iterations": 2,
                "alpha": 1.0,
                "weighting": "none",
            }
        ]
    grid: list[dict[str, Any]] = []
    for factors in (32, 64, 128):
        for regularization in (0.01, 0.05, 0.5):
            for iterations in (15, 30):
                for alpha in (1.0, 40.0):
                    for weighting in ("none", "bm25", "tfidf", "log_play"):
                        grid.append(
                            {
                                "factors": factors,
                                "regularization": regularization,
                                "iterations": iterations,
                                "alpha": alpha,
                                "weighting": weighting,
                            }
                        )
    return grid


def itemknn_grid(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [{"kind": "cosine", "neighbors": 5}]
    grid: list[dict[str, Any]] = []
    for kind in ("cosine", "bm25", "tfidf"):
        for neighbors in (20, 50, 100, 200):
            grid.append({"kind": kind, "neighbors": neighbors})
    return grid


def ease_grid(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [{"lambda": 10.0, "weighting": "none"}]
    grid: list[dict[str, Any]] = []
    for weighting in ("none", "bm25"):
        for reg_lambda in (1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0):
            grid.append({"lambda": reg_lambda, "weighting": weighting})
    return grid


def _better(candidate: float, best: float, candidate_recall: float, best_recall: float) -> bool:
    if candidate > best + 1e-12:
        return True
    if abs(candidate - best) <= 1e-12 and candidate_recall > best_recall:
        return True
    return False


def _search_family(
    trials: Sequence[dict[str, Any]],
    fit_and_score: Callable[[dict[str, Any]], dict[str, float]],
    family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_cfg: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -1.0
    best_recall = -1.0
    records: list[dict[str, Any]] = []
    for trial_index, cfg in enumerate(trials, start=1):
        metrics = fit_and_score(cfg)
        score = selection_score(metrics)
        record = {"config": cfg, "validation": metrics, "objective": round(score, 6)}
        records.append(record)
        print(
            f"val {family} {trial_index}/{len(trials)} {cfg} {metrics} obj={score:.6f}", flush=True
        )
        if best_metrics is None or _better(score, best_score, metrics["recall@10"], best_recall):
            best_cfg = cfg
            best_metrics = metrics
            best_score = score
            best_recall = metrics["recall@10"]
    if best_cfg is None or best_metrics is None:
        raise RuntimeError(f"no trials for {family}")
    return {
        "config": best_cfg,
        "validation": best_metrics,
        "objective": round(best_score, 6),
    }, records


def _blend_scores(
    cf_scores: np.ndarray, pop_scores: np.ndarray, weight: float, kind: str
) -> np.ndarray:
    if weight <= 0.0:
        return np.broadcast_to(np.asarray(pop_scores, dtype=np.float64), cf_scores.shape).copy()
    collaborative = np.nan_to_num(
        np.asarray(cf_scores, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6
    )
    if weight >= 1.0:
        return collaborative
    if kind == "zscore":
        cf_mean = collaborative.mean(axis=1, keepdims=True)
        cf_std = collaborative.std(axis=1, keepdims=True)
        cf_std = np.where(cf_std < 1e-6, 1.0, cf_std)
        cf_z = (collaborative - cf_mean) / cf_std
        pop_std = float(pop_scores.std())
        if pop_std < 1e-6:
            pop_std = 1.0
        pop_z = (pop_scores - float(pop_scores.mean())) / pop_std
        blended = (1.0 - weight) * pop_z + weight * cf_z
        return cast(np.ndarray, blended)
    if kind == "rrf":
        k = 60.0
        cf_rank = np.argsort(np.argsort(-collaborative, axis=1), axis=1).astype(np.float32)
        order = np.argsort(-pop_scores)
        pop_rank = np.empty(len(pop_scores), dtype=np.float32)
        pop_rank[order] = np.arange(len(pop_scores), dtype=np.float32)
        blended = weight / (k + cf_rank) + (1.0 - weight) / (k + pop_rank)
        return cast(np.ndarray, blended)
    raise ValueError(kind)


def _cache_scores(model: Any, ctx: EvalContext) -> np.ndarray:
    parts: list[np.ndarray] = []
    batch_size = 2048
    for start in range(0, len(ctx.user_ids), batch_size):
        batch = ctx.user_ids[start : start + batch_size]
        parts.append(model.score_batch(batch, ctx.catalog))
    return np.vstack(parts)


def tune_retrievers(organic: pd.DataFrame, seed: int, smoke: bool) -> dict[str, Any]:
    if "date" not in organic.columns:
        raise ValueError("organic log must include date for the validation split")
    fit_df = organic.loc[organic["date"] < VAL_START_DATE].reset_index(drop=True)
    val_df = organic.loc[organic["date"] >= VAL_START_DATE].reset_index(drop=True)
    if fit_df.empty or val_df.empty:
        raise ValueError("organic validation split produced an empty slice")
    ctx = build_eval_context(fit_df, val_df)
    print(
        f"validation fit_rows={len(fit_df)} val_rows={len(val_df)} "
        f"val_users={len(ctx.user_ids)} catalog={len(ctx.catalog)}",
        flush=True,
    )
    popularity = metrics_for_scores(
        lambda batch: np.broadcast_to(ctx.pop_scores, (len(batch), len(ctx.catalog))).copy(), ctx
    )
    print(f"val popularity {popularity} obj={selection_score(popularity):.6f}", flush=True)

    def score_model(model: Any) -> dict[str, float]:
        return metrics_for_scores(lambda batch: model.score_batch(batch, ctx.catalog), ctx)

    als_chosen, als_trials = _search_family(
        als_grid(smoke),
        lambda cfg: score_model(
            fit_als(
                fit_df,
                factors=int(cfg["factors"]),
                regularization=float(cfg["regularization"]),
                iterations=int(cfg["iterations"]),
                alpha=float(cfg["alpha"]),
                weighting=str(cfg["weighting"]),
                seed=seed,
            )
        ),
        "als",
    )
    knn_chosen, knn_trials = _search_family(
        itemknn_grid(smoke),
        lambda cfg: score_model(
            fit_itemknn(fit_df, kind=str(cfg["kind"]), neighbors=int(cfg["neighbors"]))
        ),
        "itemknn",
    )
    ease_chosen, ease_trials = _search_family(
        ease_grid(smoke),
        lambda cfg: score_model(
            fit_ease(fit_df, reg_lambda=float(cfg["lambda"]), weighting=str(cfg["weighting"]))
        ),
        "ease",
    )

    refit = {
        "als_tuned": fit_als(
            fit_df,
            factors=int(als_chosen["config"]["factors"]),
            regularization=float(als_chosen["config"]["regularization"]),
            iterations=int(als_chosen["config"]["iterations"]),
            alpha=float(als_chosen["config"]["alpha"]),
            weighting=str(als_chosen["config"]["weighting"]),
            seed=seed,
        ),
        "itemknn": fit_itemknn(
            fit_df,
            kind=str(knn_chosen["config"]["kind"]),
            neighbors=int(knn_chosen["config"]["neighbors"]),
        ),
        "ease": fit_ease(
            fit_df,
            reg_lambda=float(ease_chosen["config"]["lambda"]),
            weighting=str(ease_chosen["config"]["weighting"]),
        ),
    }
    cached = {name: _cache_scores(model, ctx) for name, model in refit.items()}
    weights = (
        [0.0, 1.0] if smoke else [round(float(value), 2) for value in np.linspace(0.0, 1.0, 11)]
    )
    blend_kinds = ("zscore",) if smoke else ("zscore", "rrf")
    hybrid_trials: list[dict[str, Any]] = []
    best_hybrid: dict[str, Any] | None = None
    best_score = -1.0
    best_recall = -1.0
    user_pos = {user_id: i for i, user_id in enumerate(ctx.user_ids)}
    for base_name, cf_scores in cached.items():
        for kind in blend_kinds:
            for weight in weights:
                blended = _blend_scores(cf_scores, ctx.pop_scores, float(weight), kind)

                def take(batch: list[int], blended: np.ndarray = blended) -> np.ndarray:
                    rows = [user_pos[user_id] for user_id in batch]
                    return blended[rows]

                metrics = metrics_for_scores(take, ctx)
                score = selection_score(metrics)
                record = {
                    "config": {"base": base_name, "blend": kind, "weight": float(weight)},
                    "validation": metrics,
                    "objective": round(score, 6),
                }
                hybrid_trials.append(record)
                if best_hybrid is None or _better(
                    score, best_score, metrics["recall@10"], best_recall
                ):
                    best_hybrid = record
                    best_score = score
                    best_recall = metrics["recall@10"]
                print(
                    f"val hybrid {record['config']} {metrics} obj={score:.6f}",
                    flush=True,
                )
    if best_hybrid is None:
        raise RuntimeError("hybrid search produced no trials")

    families = {
        "als_tuned": als_chosen,
        "itemknn": knn_chosen,
        "ease": ease_chosen,
        "hybrid": best_hybrid,
    }
    pure = {name: families[name] for name in ("als_tuned", "itemknn", "ease")}
    best_pure = max(
        pure.items(), key=lambda item: (item[1]["objective"], item[1]["validation"]["recall@10"])
    )
    return {
        "validation_start_date": VAL_START_DATE,
        "objective": "mean(recall@10, recall@20, recall@50, ndcg@50)",
        "fit_rows": int(len(fit_df)),
        "val_rows": int(len(val_df)),
        "n_val_users": int(len(ctx.user_ids)),
        "n_val_items": int(len(ctx.catalog)),
        "popularity": {
            "validation": popularity,
            "objective": round(selection_score(popularity), 6),
        },
        "chosen": families,
        "ranker_retriever": best_pure[0],
        "trials": {
            "als": als_trials,
            "itemknn": knn_trials,
            "ease": ease_trials,
            "hybrid": hybrid_trials,
        },
    }
