from libs.schemas.recommend import RecommendResponse


def test_recommend_response_new_fields() -> None:
    obj = RecommendResponse.model_validate(
        {
            "request_id": "r1",
            "user_id": 1,
            "items": [
                {
                    "item_id": 123,
                    "rank": 1,
                    "final_score": 0.75,
                    "objective_scores": {
                        "click": 0.1,
                        "long_view": 0.2,
                        "watch_time": 0.3,
                    },
                    "blend_features": {
                        "als_score": 0.6,
                        "bpr_score": 0.5,
                        "two_tower_score": 0.4,
                    },
                }
            ],
            "model_versions": {
                "als": "als_v2",
                "bpr": "bpr_v1",
                "two_tower": "twotower_v1",
                "rank_click": "ranker_v1",
                "rank_long_view": "ranker_v1",
                "rank_watch_time": "ranker_v1",
                "blend_meta": "ranker_v1",
            },
            "latency_ms": {"feature_fetch": 1.0, "retrieval": 2.0, "ranking": 3.0, "total": 6.0},
            "fallback_used": None,
            "candidate_count": 250,
            "experiment_assignments": {"exp_watch_v2": "control"},
        }
    )
    assert obj.items[0].final_score > 0
    assert obj.candidate_count == 250
