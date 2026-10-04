import math

import numpy as np

from pipelines.evaluation.metrics import macro_average, ndcg_at_k, recall_at_k


def test_recall_at_k_partial_hit() -> None:
    ranked = np.array([0, 2, 1, 3])
    assert recall_at_k(ranked, {0, 1}, k=2) == 0.5
    assert recall_at_k(ranked, {0, 1}, k=3) == 1.0


def test_recall_empty_relevant_is_zero() -> None:
    assert recall_at_k(np.array([1, 2, 3]), set(), k=2) == 0.0


def test_ndcg_perfect_and_partial() -> None:
    perfect = np.array([0, 1, 2])
    assert ndcg_at_k(perfect, {0, 1}, k=2) == 1.0

    ranked = np.array([0, 2, 1])
    dcg = 1.0 / math.log2(2)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, {0, 1}, k=2) == dcg / idcg


def test_macro_average() -> None:
    assert macro_average([0.0, 1.0]) == 0.5
    assert macro_average([]) == 0.0
