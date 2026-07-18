import json
from datetime import datetime, timedelta, timezone

from app.domain.nrim.simulation.model_ready_exporter import (
    export_scenario_windows,
)
from app.domain.nrim.simulation.temporal_windowing import (
    TemporalWindowSpecification,
)


def write_json(path, value) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(value, file, indent=2)


def test_exporter_creates_model_ready_windows(
    tmp_path,
) -> None:
    start = datetime(
        2026,
        7,
        18,
        tzinfo=timezone.utc,
    )

    telemetry = []

    for index in range(13):
        timestamp = (
            start + timedelta(hours=index)
        )

        telemetry.append(
            {
                "event_id": f"event_{index}",
                "deployment_id": "deployment_1",
                "component_node_id": "storage_1",
                "observed_at": timestamp.isoformat(),
                "ingested_at": timestamp.isoformat(),
                "signal_type": "metric",
                "signal_name": (
                    "system.disk.utilization"
                ),
                "value": 60.0 + index,
                "unit": "percent",
                "attributes": [],
                "collection_source": "simulator",
                "quality": "high",
                "baseline_status": "unknown",
                "service_node_ids": [],
                "evidence_ids": [],
            }
        )

    observable = {
        "scenario_id": "scenario_abcdef",
        "environment_id": "environment_1",
        "pair_id": "pair_1",
        "simulation_metadata": {
            "confounder_count": 0
        },
        "topology": {
            "nodes": [
                {
                    "node_id": "storage_1",
                    "node_type": "catchup_storage",
                }
            ],
            "edges": [],
        },
        "telemetry": telemetry,
        "context_events": [],
    }

    hidden = {
        "ground_truth": {
            "failure_type": "healthy",
            "root_cause_node_id": None,
            "incident_onset_time": None,
            "affected_service_node_ids": [],
        }
    }

    observable_path = (
        tmp_path / "observable.json"
    )
    hidden_path = tmp_path / "hidden.json"

    write_json(observable_path, observable)
    write_json(hidden_path, hidden)

    records = export_scenario_windows(
        observable_path=observable_path,
        hidden_path=hidden_path,
        split="train",
        output_dir=tmp_path / "out",
        specification=(
            TemporalWindowSpecification(
                observation_hours=6,
                prediction_horizon_hours=4,
                stride_hours=2,
                minimum_event_count=2,
            )
        ),
    )

    assert records

    exported = json.loads(
        (
            tmp_path
            / "out"
            / f"{records[0].window_id}.json"
        ).read_text(encoding="utf-8")
    )

    serialized = str(exported)

    assert "environment_id" not in serialized
    assert "pair_id" not in serialized
    assert "simulation_metadata" not in serialized
