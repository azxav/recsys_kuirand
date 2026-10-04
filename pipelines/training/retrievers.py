"""Candidate generators for the KuaiRand offline eval.

ALS, item-item kNN, and EASE are fit on clicked pairs only. Score matrices
are aligned to a shared catalog so every model is ranked the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
    tfidf_weight,
)
from scipy import sparse
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

OLD_ALS: dict[str, Any] = {
    "factors": 64,
    "regularization": 0.05,
    "iterations": 15,
    "alpha": 40.0,
    "weighting": "none",
}


def _clicked(train: pd.DataFrame) -> pd.DataFrame:
    clicked = train.loc[train["is_click"] == 1, ["user_id", "video_id"]].drop_duplicates()
    if clicked.empty:
        raise ValueError("no clicked pairs")
    return clicked


def _index_maps(
    clicked: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
    users = np.sort(clicked["user_id"].unique())
    items = np.sort(clicked["video_id"].unique())
    user_index = {int(user_id): i for i, user_id in enumerate(users)}
    item_index = {int(item_id): i for i, item_id in enumerate(items)}
    return user_index, item_index, users, items


def _binary_matrix(
    train: pd.DataFrame,
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    clicked = _clicked(train)
    user_index, item_index, users, items = _index_maps(clicked)
    rows = clicked["user_id"].map(user_index).to_numpy()
    cols = clicked["video_id"].map(item_index).to_numpy()
    matrix = csr_matrix(
        (np.ones(len(clicked), dtype=np.float32), (rows, cols)),
        shape=(len(users), len(items)),
    )
    return matrix, user_index, item_index


def _weighted_matrix(
    train: pd.DataFrame,
    weighting: str,
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    if weighting == "log_play":
        if "play_time_ms" not in train.columns:
            raise ValueError("log_play weighting requires play_time_ms")
        clicked = train.loc[train["is_click"] == 1, ["user_id", "video_id", "play_time_ms"]]
        grouped = clicked.groupby(["user_id", "video_id"], as_index=False)["play_time_ms"].sum()
        user_index, item_index, users, items = _index_maps(grouped)
        values = np.log1p(grouped["play_time_ms"].clip(lower=0).to_numpy() / 1000.0)
        values = np.clip(values, 0.01, None).astype(np.float32)
        rows = grouped["user_id"].map(user_index).to_numpy()
        cols = grouped["video_id"].map(item_index).to_numpy()
        matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
        return matrix, user_index, item_index

    matrix, user_index, item_index = _binary_matrix(train)
    if weighting == "none":
        return matrix, user_index, item_index
    if weighting == "bm25":
        # Defaults recommended by implicit for ALS preprocessing.
        weighted = bm25_weight(matrix, K1=100, B=0.8).tocsr()
        return weighted, user_index, item_index
    if weighting == "tfidf":
        weighted = tfidf_weight(matrix).tocsr()
        return weighted, user_index, item_index
    raise ValueError(f"unknown weighting {weighting}")


@dataclass
class FactorModel:
    user_index: dict[int, int]
    item_index: dict[int, int]
    user_factors: np.ndarray
    item_factors: np.ndarray

    def score_batch(self, user_ids: list[int], catalog: np.ndarray) -> np.ndarray:
        catalog_index = {int(item_id): i for i, item_id in enumerate(catalog)}
        aligned = np.zeros((len(catalog), self.item_factors.shape[1]), dtype=np.float32)
        for raw_id, row in self.item_index.items():
            dest = catalog_index.get(raw_id)
            if dest is not None:
                aligned[dest] = self.item_factors[row]
        rows = [self.user_index[user_id] for user_id in user_ids]
        return self.user_factors[rows] @ aligned.T

    def pair_scores(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(user_ids), dtype=np.float32)
        user_codes = pd.Series(user_ids).map(self.user_index)
        item_codes = pd.Series(item_ids).map(self.item_index)
        known = (user_codes.notna() & item_codes.notna()).to_numpy()
        if known.any():
            u_rows = user_codes[known].to_numpy(dtype=np.int32)
            i_rows = item_codes[known].to_numpy(dtype=np.int32)
            dots = np.sum(self.user_factors[u_rows] * self.item_factors[i_rows], axis=1)
            scores[known] = dots.astype(np.float32, copy=False)
        return scores


@dataclass
class SimilarityModel:
    """Item-item kNN or EASE. ``item_item`` is items x items in ``item_index`` order."""

    user_index: dict[int, int]
    item_index: dict[int, int]
    user_items: csr_matrix
    item_item: sparse.spmatrix | np.ndarray
    _cached_scores: np.ndarray | None = field(default=None, repr=False)

    def full_scores(self) -> np.ndarray:
        if self._cached_scores is None:
            raw = self.user_items @ self.item_item
            if sparse.issparse(raw):
                raw = raw.toarray()
            self._cached_scores = np.asarray(raw, dtype=np.float32)
        return self._cached_scores

    def score_batch(self, user_ids: list[int], catalog: np.ndarray) -> np.ndarray:
        rows = [self.user_index[user_id] for user_id in user_ids]
        raw = self.user_items[rows] @ self.item_item
        if sparse.issparse(raw):
            raw = raw.toarray()
        raw = np.asarray(raw, dtype=np.float32)
        catalog_index = {int(item_id): i for i, item_id in enumerate(catalog)}
        cols = []
        keep = []
        for raw_id, row in self.item_index.items():
            dest = catalog_index.get(raw_id)
            if dest is not None:
                keep.append(row)
                cols.append(dest)
        out = np.zeros((len(user_ids), len(catalog)), dtype=np.float32)
        if cols:
            out[:, np.asarray(cols, dtype=np.int32)] = raw[:, np.asarray(keep, dtype=np.int32)]
        return out

    def pair_scores(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        raw = self.full_scores()
        scores = np.zeros(len(user_ids), dtype=np.float32)
        user_codes = pd.Series(user_ids).map(self.user_index)
        item_codes = pd.Series(item_ids).map(self.item_index)
        known = (user_codes.notna() & item_codes.notna()).to_numpy()
        if known.any():
            u_rows = user_codes[known].to_numpy(dtype=np.int32)
            i_rows = item_codes[known].to_numpy(dtype=np.int32)
            scores[known] = raw[u_rows, i_rows]
        return scores


def fit_als(
    train: pd.DataFrame,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    weighting: str,
    seed: int,
) -> FactorModel:
    matrix, user_index, item_index = _weighted_matrix(train, weighting)
    threadpool_limits(limits=1, user_api="blas")
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
        num_threads=0,
    )
    model.fit(matrix, show_progress=False)
    return FactorModel(
        user_index=user_index,
        item_index=item_index,
        user_factors=np.asarray(model.user_factors),
        item_factors=np.asarray(model.item_factors),
    )


def fit_itemknn(train: pd.DataFrame, kind: str, neighbors: int) -> SimilarityModel:
    matrix, user_index, item_index = _binary_matrix(train)
    if kind == "cosine":
        model = CosineRecommender(K=neighbors, num_threads=0)
    elif kind == "bm25":
        model = BM25Recommender(K=neighbors, num_threads=0)
    elif kind == "tfidf":
        model = TFIDFRecommender(K=neighbors, num_threads=0)
    else:
        raise ValueError(f"unknown itemknn kind {kind}")
    model.fit(matrix, show_progress=False)
    if model.similarity is None:
        raise RuntimeError("itemknn fit did not produce a similarity matrix")
    return SimilarityModel(
        user_index=user_index,
        item_index=item_index,
        user_items=matrix,
        item_item=model.similarity.tocsr(),
    )


def fit_ease(train: pd.DataFrame, reg_lambda: float, weighting: str) -> SimilarityModel:
    matrix, user_index, item_index = _weighted_matrix(train, weighting)
    gram = (matrix.T @ matrix).toarray().astype(np.float64, copy=False)
    item_item = ease_coefficients(gram, reg_lambda)
    # Pair scores use the same matrix that built the gram.
    return SimilarityModel(
        user_index=user_index,
        item_index=item_index,
        user_items=matrix.tocsr(),
        item_item=item_item,
    )


def ease_coefficients(gram: np.ndarray, reg_lambda: float) -> np.ndarray:
    """Steck EASE: B_ij = -P_ij / P_jj with a zero diagonal, P = (G + λI)^-1."""
    regularized = gram.copy()
    width = regularized.shape[0]
    regularized.flat[:: width + 1] += reg_lambda
    precision = np.linalg.inv(regularized)
    diagonal = np.diag(precision).copy()
    diagonal[np.abs(diagonal) < 1e-12] = 1e-12
    coefficients = (-precision / diagonal).astype(np.float32)
    np.fill_diagonal(coefficients, 0.0)
    return cast(np.ndarray, coefficients)
