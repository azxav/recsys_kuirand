from services.api.experiments import pick_variant


def test_pick_variant_is_deterministic() -> None:
    variants = {"control": 0.5, "treatment": 0.5}
    first = pick_variant("exp_watch", 12345, variants)
    second = pick_variant("exp_watch", 12345, variants)
    assert first == second


def test_pick_variant_returns_existing_variant() -> None:
    variants = {"a": 0.1, "b": 0.3, "c": 0.6}
    assigned = {pick_variant("exp_mix", uid, variants) for uid in range(1, 200)}
    assert assigned.issubset(set(variants))
