from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from libs.common.config import get_settings


@dataclass
class Candidate:
    item_id: int
    als_score: float
    bpr_score: float
    two_tower_score: float


class ObjectiveScores(TypedDict):
    click: float
    long_view: float
    watch_time: float


class BlendFeatures(TypedDict):
    als_score: float
    bpr_score: float
    two_tower_score: float


class RankedCandidate(TypedDict):
    item_id: int
    final_score: float
    objective_scores: ObjectiveScores
    blend_features: BlendFeatures


class RetrievalStore:
    def __init__(
        self,
        als_dir: Path,
        bpr_dir: Path,
        two_tower_dir: Path,
        enable_two_tower: bool,
    ) -> None:
        self.als_dir = als_dir
        self.bpr_dir = bpr_dir
        self.two_tower_dir = two_tower_dir
        self.enable_two_tower = enable_two_tower

        self.als_item_factors: np.ndarray | None = None
        self.als_user_factors: np.ndarray | None = None
        self.als_item_index: dict[int, int] = {}
        self.als_user_index: dict[int, int] = {}

        self.bpr_item_factors: np.ndarray | None = None
        self.bpr_user_factors: np.ndarray | None = None
        self.bpr_item_index: dict[int, int] = {}
        self.bpr_user_index: dict[int, int] = {}

        self.two_tower_item_factors: np.ndarray | None = None
        self.two_tower_user_factors: np.ndarray | None = None
        self.two_tower_item_index: dict[int, int] = {}
        self.two_tower_user_index: dict[int, int] = {}

        self._load_all()

    def _read_index(self, path: Path) -> dict[int, int]:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}

    def _load_factor_bundle(
        self,
        base_dir: Path,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, int], dict[int, int]]:
        item_path = base_dir / "item_factors.npy"
        user_path = base_dir / "user_factors.npy"
        if not (item_path.exists() and user_path.exists()):
            return None, None, {}, {}
        item_factors = np.load(item_path)
        user_factors = np.load(user_path)
        item_index = self._read_index(base_dir / "item_index.json")
        user_index = self._read_index(base_dir / "user_index.json")
        return item_factors, user_factors, item_index, user_index

    def _load_all(self) -> None:
        (
            self.als_item_factors,
            self.als_user_factors,
            self.als_item_index,
            self.als_user_index,
        ) = self._load_factor_bundle(self.als_dir)

        (
            self.bpr_item_factors,
            self.bpr_user_factors,
            self.bpr_item_index,
            self.bpr_user_index,
        ) = self._load_factor_bundle(self.bpr_dir)

        if self.enable_two_tower:
            (
                self.two_tower_item_factors,
                self.two_tower_user_factors,
                self.two_tower_item_index,
                self.two_tower_user_index,
            ) = self._load_factor_bundle(self.two_tower_dir)

    def _topk_scores(self, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        if scores.size == 0:
            return []
        k = min(k, scores.size)
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(int(i), float(scores[i])) for i in idx]

    def _model_candidates(
        self,
        user_id: int,
        k: int,
        user_factors: np.ndarray | None,
        item_factors: np.ndarray | None,
        user_index: dict[int, int],
        item_index: dict[int, int],
    ) -> dict[int, float]:
        if user_factors is None or item_factors is None:
            return {}
        user_idx = user_index.get(user_id)
        if user_idx is None:
            return {}
        user_vec = user_factors[user_idx]
        scores = item_factors @ user_vec
        top = self._topk_scores(scores, k)
        return {item_index[i]: s for i, s in top if i in item_index}

    def als_candidates(self, user_id: int, k: int) -> dict[int, float]:
        return self._model_candidates(
            user_id,
            k,
            self.als_user_factors,
            self.als_item_factors,
            self.als_user_index,
            self.als_item_index,
        )

    def bpr_candidates(self, user_id: int, k: int) -> dict[int, float]:
        return self._model_candidates(
            user_id,
            k,
            self.bpr_user_factors,
            self.bpr_item_factors,
            self.bpr_user_index,
            self.bpr_item_index,
        )

    def two_tower_candidates(self, user_id: int, k: int) -> dict[int, float]:
        return self._model_candidates(
            user_id,
            k,
            self.two_tower_user_factors,
            self.two_tower_item_factors,
            self.two_tower_user_index,
            self.two_tower_item_index,
        )


class RankerStore:
    def __init__(self, ranker_dir: Path) -> None:
        self.ranker_dir = ranker_dir

        self.click_model: CatBoostClassifier | None = None
        self.long_view_model: CatBoostClassifier | None = None
        self.watch_model: CatBoostRegressor | None = None
        self.blend_model: CatBoostRegressor | None = None

        self._load_all()

    def _load_classifier(self, path: Path) -> CatBoostClassifier | None:
        if not path.exists():
            return None
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model

    def _load_regressor(self, path: Path) -> CatBoostRegressor | None:
        if not path.exists():
            return None
        model = CatBoostRegressor()
        model.load_model(str(path))
        return model

    def _load_all(self) -> None:
        self.click_model = self._load_classifier(self.ranker_dir / "click_model.cbm")
        self.long_view_model = self._load_classifier(self.ranker_dir / "long_view_model.cbm")
        self.watch_model = self._load_regressor(self.ranker_dir / "watch_time_model.cbm")
        self.blend_model = self._load_regressor(self.ranker_dir / "blend_model.cbm")

    def ready(self) -> bool:
        return all([self.click_model, self.long_view_model, self.watch_model, self.blend_model])


@lru_cache(maxsize=1)
def get_retrieval_store() -> RetrievalStore:
    s = get_settings()
    return RetrievalStore(
        Path(s.als_artifact_dir),
        Path(s.bpr_artifact_dir),
        Path(s.two_tower_artifact_dir),
        s.enable_two_tower,
    )


@lru_cache(maxsize=1)
def get_ranker_store() -> RankerStore:
    s = get_settings()
    return RankerStore(Path(s.ranker_artifact_dir))


def merge_candidates(user_id: int, k: int) -> list[Candidate]:
    s = get_settings()
    store = get_retrieval_store()
    als_scores = store.als_candidates(user_id, k)
    bpr_scores = store.bpr_candidates(user_id, k)
    two_tower_scores = store.two_tower_candidates(user_id, k) if s.enable_two_tower else {}

    merged: dict[int, Candidate] = {}
    for item_id, score in als_scores.items():
        merged[item_id] = Candidate(
            item_id=item_id,
            als_score=float(score),
            bpr_score=0.0,
            two_tower_score=0.0,
        )
    for item_id, score in bpr_scores.items():
        if item_id in merged:
            merged[item_id].bpr_score = float(score)
        else:
            merged[item_id] = Candidate(
                item_id=item_id,
                als_score=0.0,
                bpr_score=float(score),
                two_tower_score=0.0,
            )
    for item_id, score in two_tower_scores.items():
        if item_id in merged:
            merged[item_id].two_tower_score = float(score)
        else:
            merged[item_id] = Candidate(
                item_id=item_id,
                als_score=0.0,
                bpr_score=0.0,
                two_tower_score=float(score),
            )
    return list(merged.values())


def _fallback_rank(candidates: list[Candidate]) -> list[RankedCandidate]:
    s = get_settings()
    fallback_rows: list[RankedCandidate] = []
    for c in candidates:
        final = float(
            s.als_weight * c.als_score
            + s.bpr_weight * c.bpr_score
            + s.two_tower_weight * c.two_tower_score
        )
        fallback_rows.append(
            {
                "item_id": c.item_id,
                "final_score": final,
                "objective_scores": {
                    "click": 0.0,
                    "long_view": 0.0,
                    "watch_time": 0.0,
                },
                "blend_features": {
                    "als_score": float(c.als_score),
                    "bpr_score": float(c.bpr_score),
                    "two_tower_score": float(c.two_tower_score),
                },
            }
        )
    return fallback_rows


def rank_candidates(candidates: list[Candidate], tab: int | None) -> list[RankedCandidate]:
    ranker = get_ranker_store()
    tab_val = float(tab or 0)

    if not candidates:
        return []

    feature_rows = np.array(
        [
            [
                c.als_score,
                c.bpr_score,
                c.two_tower_score,
                max(c.als_score, c.bpr_score, c.two_tower_score),
                min(c.als_score, c.bpr_score, c.two_tower_score),
                tab_val,
            ]
            for c in candidates
        ],
        dtype=np.float32,
    )

    if not ranker.ready():
        return _fallback_rank(candidates)

    assert ranker.click_model is not None
    assert ranker.long_view_model is not None
    assert ranker.watch_model is not None
    assert ranker.blend_model is not None

    try:
        click = ranker.click_model.predict_proba(feature_rows)[:, 1]
        long_view = ranker.long_view_model.predict_proba(feature_rows)[:, 1]
        watch = ranker.watch_model.predict(feature_rows)

        blend_features = np.column_stack(
            [
                click,
                long_view,
                watch,
                feature_rows[:, 0],
                feature_rows[:, 1],
                feature_rows[:, 2],
                feature_rows[:, 5],
            ]
        )
        final_scores = ranker.blend_model.predict(blend_features)
    except Exception:
        return _fallback_rank(candidates)

    ranked_rows: list[RankedCandidate] = []
    for i, c in enumerate(candidates):
        ranked_rows.append(
            {
                "item_id": c.item_id,
                "final_score": float(final_scores[i]),
                "objective_scores": {
                    "click": float(click[i]),
                    "long_view": float(long_view[i]),
                    "watch_time": float(watch[i]),
                },
                "blend_features": {
                    "als_score": float(c.als_score),
                    "bpr_score": float(c.bpr_score),
                    "two_tower_score": float(c.two_tower_score),
                },
            }
        )
    return ranked_rows
