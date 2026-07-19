import json

from app.domain.nrim.benchmark.dataset_fingerprint import (
    build_dataset_fingerprint,
    sha256_json,
    verify_dataset_fingerprint,
)


def make_manifest(tmp_path) -> dict:
    window_path = tmp_path / "window_1.json"

    window_path.write_text(
        json.dumps(
            {
                "window_id": "window_1",
                "features": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )

    return {
        "dataset_name": "test",
        "dataset_version": "0.1.0",
        "source_dataset_version": "0.1.0",
        "window_specification": {
            "observation_hours": 6,
            "prediction_horizon_hours": 6,
        },
        "feature_schema": {
            "node_features": [],
            "edge_features": [],
        },
        "records": [
            {
                "window_id": "window_1",
                "path": str(window_path),
                "split": "train",
            }
        ],
    }


def test_json_hash_is_order_independent() -> None:
    first = sha256_json(
        {
            "a": 1,
            "b": 2,
        }
    )

    second = sha256_json(
        {
            "b": 2,
            "a": 1,
        }
    )

    assert first == second


def test_fingerprint_is_reproducible(tmp_path) -> None:
    manifest = make_manifest(tmp_path)

    first = build_dataset_fingerprint(
        manifest=manifest
    )
    second = build_dataset_fingerprint(
        manifest=manifest
    )

    assert first == second


def test_changed_window_invalidates_fingerprint(
    tmp_path,
) -> None:
    manifest = make_manifest(tmp_path)

    expected = build_dataset_fingerprint(
        manifest=manifest
    )

    window_path = tmp_path / "window_1.json"

    window_path.write_text(
        json.dumps(
            {
                "window_id": "window_1",
                "features": [9, 9, 9],
            }
        ),
        encoding="utf-8",
    )

    errors = verify_dataset_fingerprint(
        manifest=manifest,
        expected=expected,
    )

    assert errors
