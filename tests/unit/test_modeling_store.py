from services.api.modeling import merge_candidates, rank_candidates


def test_modeling_fallback_without_artifacts() -> None:
    candidates = merge_candidates(user_id=999999, k=50)
    assert candidates == []

    scored = rank_candidates([], tab=1)
    assert scored == []
