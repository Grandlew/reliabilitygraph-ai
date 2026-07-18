from app.domain.nrim.simulation.model_ready_audit import (
    audit_window_record,
    recursively_find_forbidden_keys,
)


def test_audit_detects_pair_id() -> None:
    errors = recursively_find_forbidden_keys(
        {
            "features": {
                "pair_id": "pair_1"
            }
        }
    )

    assert errors


def test_audit_accepts_basic_clean_window() -> None:
    record = {
        "window_id": "window_123abc",
        "source_scenario_id": "scenario_abcd",
        "observation_start": (
            "2026-07-18T00:00:00+00:00"
        ),
        "observation_cutoff": (
            "2026-07-18T06:00:00+00:00"
        ),
        "prediction_end": (
            "2026-07-18T12:00:00+00:00"
        ),
        "node_ids": [
            "storage_1",
            "catchup_1",
        ],
        "node_feature_names": [
            "feature_1"
        ],
        "node_features": [
            [0.0],
            [1.0],
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
            "root_cause_node": [1, 0],
            "affected_service_node": [0, 1],
        },
    }

    errors = audit_window_record(record)

    assert errors == []


def test_window_identifier_cannot_encode_class() -> None:
    record = {
        "window_id": (
            "window_storage_failure_1"
        ),
        "node_ids": [],
        "node_features": [],
        "edge_index": [],
        "edge_features": [],
        "targets": {
            "root_cause_node": [],
            "affected_service_node": [],
        },
    }

    errors = audit_window_record(record)

    assert errors
