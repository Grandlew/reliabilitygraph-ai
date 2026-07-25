import json
from pathlib import Path

import pytest

from app.domain.nrim.benchmark.dataset_loader import (
    iter_window_records,
    load_model_ready_manifest,
    missing_feature_fraction,
    resolve_window_record_path,
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


def test_missing_fraction_excludes_non_applicable_signals() -> None:
    result = missing_feature_fraction(
        feature_names=[
            "value__applicable",
            "value__missing",
        ],
        matrix=[
            [1.0, 1.0],
            [0.0, 0.0],
        ],
    )

    assert result == 1.0


@pytest.mark.parametrize(
    "recorded_path",
    [
        r"C:\retired\day09_model_ready_v06\train\window_abc.json",
        "/retired/day09_model_ready_v06/train/window_abc.json",
    ],
)
def test_model_ready_loader_uses_colocated_window(
    tmp_path: Path,
    recorded_path: str,
) -> None:
    window_dir = tmp_path / "train"
    window_dir.mkdir()
    (window_dir / "window_abc.json").write_text(
        json.dumps(make_window()),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "window_id": "window_abc",
                        "split": "train",
                        "path": recorded_path,
                    }
                ],
                "feature_schema": {},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_model_ready_manifest(manifest_path)
    records = list(iter_window_records(manifest))

    assert records[0][1]["window_id"] == "window_abc"
    assert not Path(recorded_path).exists()


@pytest.mark.parametrize(
    "recorded_path",
    [
        "../window_abc.json",
        r"C:\retired\train\wrong.json",
    ],
)
def test_model_ready_resolver_rejects_untrusted_recorded_path(
    tmp_path: Path,
    recorded_path: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [],
                "feature_schema": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = load_model_ready_manifest(manifest_path)

    with pytest.raises(ValueError):
        resolve_window_record_path(
            manifest=manifest,
            record={
                "window_id": "window_abc",
                "split": "train",
                "path": recorded_path,
            },
        )
