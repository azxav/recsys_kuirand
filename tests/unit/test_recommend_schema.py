import pytest
from pydantic import ValidationError

from libs.schemas.recommend import RecommendRequest


def test_k_bounds() -> None:
    with pytest.raises(ValidationError):
        RecommendRequest(user_id=1, k=0)

    with pytest.raises(ValidationError):
        RecommendRequest(user_id=1, k=201)

    ok = RecommendRequest(user_id=1, k=20)
    assert ok.k == 20
    assert ok.experiment_keys == []
