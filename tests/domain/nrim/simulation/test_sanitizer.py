from app.domain.nrim.simulation.sanitizer import (
    sanitize_context_event,
    sanitize_observable_scenario,
    sanitize_telemetry_event,
)


def test_top_level_research_metadata_is_removed() -> None:
    scenario = {
        "scenario_id": "scenario_abc",
        "environment_id": "environment_1",
        "pair_id": "pair_1",
        "simulation_metadata": {
            "confounder_count": 1
        },
        "topology": {
            "nodes": [],
            "edges": [],
        },
        "telemetry": [],
        "context_events": [],
    }

    sanitized = sanitize_observable_scenario(
        scenario
    )

    assert "environment_id" not in sanitized
    assert "pair_id" not in sanitized
    assert "simulation_metadata" not in sanitized


def test_simulated_transient_attribute_is_removed() -> None:
    event = {
        "signal_name": "system.disk.io_latency",
        "attributes": [
            {
                "key": "simulated_transient",
                "value": True,
            },
            {
                "key": "mount_point",
                "value": "/catchup",
            },
        ],
    }

    sanitized = sanitize_telemetry_event(event)

    keys = {
        attribute["key"]
        for attribute in sanitized[
            "attributes"
        ]
    }

    assert "simulated_transient" not in keys
    assert "mount_point" in keys


def test_context_event_keeps_observable_fact() -> None:
    event = {
        "event_id": "event_1",
        "confounder_type": (
            "temporary_latency_spike"
        ),
        "model_visible": True,
        "signal_name": "system.disk.io_latency",
        "observed_at": (
            "2026-07-18T12:00:00+00:00"
        ),
        "statement": (
            "Storage latency increased temporarily."
        ),
    }

    sanitized = sanitize_context_event(event)

    assert sanitized is not None
    assert "confounder_type" not in sanitized
    assert "model_visible" not in sanitized
    assert (
        sanitized["signal_name"]
        == "system.disk.io_latency"
    )


def test_hidden_fields_are_removed_recursively() -> None:
    scenario = {
        "scenario_id": "scenario_abc",
        "topology": {
            "nodes": [
                {
                    "node_id": "storage_1",
                    "latent_error_factor": 4.0,
                }
            ],
            "edges": [],
        },
        "telemetry": [],
        "context_events": [],
    }

    sanitized = sanitize_observable_scenario(
        scenario
    )

    assert (
        "latent_error_factor"
        not in str(sanitized)
    )
