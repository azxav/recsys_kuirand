import json
from pathlib import Path


def test_committed_kuairand_metrics_file() -> None:
    path = Path("results/kuairand_pure_metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "KuaiRand-Pure"
    assert payload["label"] == "is_click"
    assert payload["retrieval"]["n_users"] > 1000
    assert payload["retrieval"]["n_items"] > 1000
    for model in ("als", "popularity"):
        metrics = payload["retrieval"]["models"][model]
        for key in ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50"):
            assert 0.0 <= metrics[key] <= 1.0
    assert payload["ranker"]["n_test"] > payload["ranker"]["n_train"] / 2
    for model in ("catboost", "popularity"):
        metrics = payload["ranker"]["models"][model]
        assert 0.0 <= metrics["auc"] <= 1.0
        assert metrics["logloss"] > 0.0
