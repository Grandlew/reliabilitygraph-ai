from app.domain.nrim.simulation.quality_audit import (
    audit_observable_scenario,
    audit_root_cause_labels,
)


def test_audit_detects_hidden_key() -> None:
    scenario = {
        "scenario_id": "scenario_123",
        "telemetry": [],
        "latent_error_factor": 4.0,
    }

    warnings = audit_observable_scenario(scenario)

    assert warnings


def test_audit_detects_class_in_scenario_id() -> None:
    scenario = {
        "scenario_id": (
            "scenario_storage_failure_001"
        ),
        "telemetry": [],
    }

    warnings = audit_observable_scenario(scenario)

    assert warnings


def test_audit_detects_failure_class_in_event_message() -> None:
    scenario = {
        "scenario_id": "scenario_123",
        "telemetry": [
            {
                "signal_type": "log",
                "value": "Detected storage_io_degradation",
            }
        ],
    }

    warnings = audit_observable_scenario(scenario)

    assert warnings


def test_healthy_scenario_requires_zero_root_causes() -> None:
    scenario = {
        "labels": {
            "failure_type": "healthy",
            "root_cause_node": {
                "node_1": 1,
                "node_2": 0,
            },
        }
    }

    errors = audit_root_cause_labels(scenario)

    assert errors
