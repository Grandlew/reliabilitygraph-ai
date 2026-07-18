import pytest
from pydantic import ValidationError

from app.domain.nrim.simulation.dataset_manifest import (
    DatasetManifest,
    ScenarioManifestRecord,
    manifest_summary,
)
from app.domain.nrim.simulation.scenario_design import (
    DatasetSplit,
)


def make_record(
    *,
    scenario_id: str,
    pair_id: str,
    healthy: bool,
) -> ScenarioManifestRecord:
    return ScenarioManifestRecord(
        scenario_id=scenario_id,
        pair_id=pair_id,
        environment_id="environment_1",
        topology_fingerprint="topology_1",
        split=DatasetSplit.TRAIN,
        failure_type=(
            "healthy"
            if healthy
            else "storage_io_degradation"
        ),
        healthy=healthy,
        room_count=100,
        floor_count=4,
        fault_severity=None if healthy else 0.7,
        fault_injection_fraction=(
            None if healthy else 0.5
        ),
        confounders=[],
        missingness_mode="none",
        missing_fraction=0.0,
        observable_path="observable.json",
        hidden_path="hidden.json",
        topology_seed=1,
        workload_seed=2,
        observation_seed=3,
        fault_seed=4,
    )


def test_manifest_rejects_duplicate_scenario_ids() -> None:
    record = make_record(
        scenario_id="scenario_1",
        pair_id="pair_1",
        healthy=True,
    )

    with pytest.raises(ValidationError):
        DatasetManifest(
            dataset_name="test",
            dataset_version="0.1.0",
            simulator_version="0.2.0",
            generation_seed=42,
            records=[record, record],
        )


def test_manifest_summary_counts_scenarios() -> None:
    manifest = DatasetManifest(
        dataset_name="test",
        dataset_version="0.1.0",
        simulator_version="0.2.0",
        generation_seed=42,
        records=[
            make_record(
                scenario_id="scenario_1",
                pair_id="pair_1",
                healthy=True,
            ),
            make_record(
                scenario_id="scenario_2",
                pair_id="pair_1",
                healthy=False,
            ),
        ],
    )

    summary = manifest_summary(manifest)

    assert summary["scenario_count"] == 2
    assert summary["healthy_count"] == 1
    assert summary["faulty_count"] == 1
