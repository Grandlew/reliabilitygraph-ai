import pytest

from app.domain.nrim.baselines.feature_access import (
    FeatureAccessor,
    true_root_cause_node_id,
)


def test_feature_accessor_returns_value() -> None:
    accessor = FeatureAccessor(
        ["first", "second"]
    )

    assert accessor.get(
        [1.0, 2.0],
        "second",
    ) == 2.0


def test_feature_accessor_uses_default() -> None:
    accessor = FeatureAccessor(["first"])

    assert accessor.get(
        [1.0],
        "missing",
        default=3.0,
    ) == 3.0


def test_healthy_window_has_no_root_cause() -> None:
    window = {
        "node_ids": ["a", "b"],
        "targets": {
            "root_cause_node": [0, 0]
        },
    }

    assert (
        true_root_cause_node_id(window)
        is None
    )


def test_multiple_root_causes_are_rejected() -> None:
    window = {
        "node_ids": ["a", "b"],
        "targets": {
            "root_cause_node": [1, 1]
        },
    }

    with pytest.raises(ValueError):
        true_root_cause_node_id(window)
