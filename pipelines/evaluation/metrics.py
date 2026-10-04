"""Ranking metrics used by the KuaiRand offline evaluation."""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked_item_ids: np.ndarray, relevant: set[int], k: int) -> float:
    """Fraction of relevant items found in the top-k of ``ranked_item_ids``."""
    if not relevant or k <= 0:
        return 0.0
    top = ranked_item_ids[:k]
    hits = sum(1 for item_id in top if int(item_id) in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_item_ids: np.ndarray, relevant: set[int], k: int) -> float:
    """Binary NDCG@k. Ideal DCG uses min(k, |relevant|) relevant items."""
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(ranked_item_ids[:k]):
        if int(item_id) in relevant:
            dcg += 1.0 / math.log2(rank + 2)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def macro_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))
