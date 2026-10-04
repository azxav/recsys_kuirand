import json
from pathlib import Path


def test_committed_kuairand_metrics_file() -> None:
    path = Path("results/kuairand_pure_metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "KuaiRand-Pure"
    assert payload["label"] == "is_click"
    assert payload["retrieval"]["n_users"] > 1000
    assert payload["retrieval"]["n_items"] > 1000
    models = payload["retrieval"]["models"]
    for model in ("popularity", "als", "als_tuned", "itemknn", "ease", "hybrid"):
        metrics = models[model]
        for key in ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50"):
            assert 0.0 <= metrics[key] <= 1.0
    # Published baseline, kept so a later eval cannot silently replace it.
    assert models["popularity"]["recall@10"] == 0.003293
    assert models["popularity"]["ndcg@50"] == 0.006181
    assert models["als"]["recall@10"] == 0.002819
    assert models["als"]["recall@50"] == 0.014008
    assert models["als"]["ndcg@50"] == 0.006347

    selection = payload["retrieval_selection"]
    assert selection["validation_start_date"] == 20220418
    assert selection["fit_rows"] > selection["val_rows"]
    for family in ("als_tuned", "itemknn", "ease", "hybrid"):
        assert "config" in selection["chosen"][family]
        assert selection["chosen"][family]["objective"] > 0.0

    full = payload["ranker"]["uncalibrated_full_train"]
    assert full["n_test"] > full["n_train"] / 2
    assert full["models"]["catboost"]["auc"] == 0.65684
    assert full["models"]["catboost"]["logloss"] == 0.664248
    assert full["models"]["popularity"]["auc"] == 0.573373
    assert full["models"]["popularity"]["logloss"] == 0.522033
    selected = payload["ranker"]["with_selected_retriever"]["models"]["catboost"]
    assert 0.0 <= selected["auc"] <= 1.0
    assert selected["logloss"] > 0.0
    for name in ("catboost_als_previous", "catboost_selected_retriever", "popularity"):
        block = payload["ranker"]["calibration"][name]
        assert block["method"] in {"isotonic", "platt"}
        assert block["after"]["logloss"] > 0.0
        assert 0.0 <= block["after"]["auc"] <= 1.0
