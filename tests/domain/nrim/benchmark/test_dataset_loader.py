import pytest

from app.domain.nrim.benchmark.dataset_loader import (
    missing_feature_fraction,
    summarize_window,
    validate_numeric_matrix,
)


def make_window() -> dict:
    return {
        "window_id": "window_abc",
        "split": "train",
        "node_ids": [
            "storage_1",
            "catchup_1",
        ],
        "node_feature_names": [
            "feature_1",
            "signal__missing",
        ],
        "node_features": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "edge_index": [
            [1, 0]
        ],
        "edge_feature_names": [
            "edge_feature_1"
        ],
        "edge_features": [
            [1.0]
        ],
        "targets": {
            "failure_type": (
                "storage_io_degradation"
            ),
            "current_incident": 0,
            "future_incident": 1,
            "root_cause_node": [1, 0],
            "affected_service_node": [0, 1],
        },
    }


def test_summarize_window() -> None:
    summary = summarize_window(
        make_window()
    )

    assert summary.node_count == 2
    assert summary.edge_count == 1
    assert (
        summary.root_cause_positive_count
        == 1
    )
    assert (
        summary.missing_feature_fraction
        == 0.5
    )


def test_numeric_matrix_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        validate_numeric_matrix(
            [["leakage"]],
            expected_columns=1,
            matrix_name="features",
        )


def test_missing_fraction_uses_masks_only() -> None:
    result = missing_feature_fraction(
        feature_names=[
            "value",
            "value__missing",
        ],
        matrix=[
            [100.0, 0.0],
            [0.0, 1.0],
        ],
    )

    assert result == 0.5
