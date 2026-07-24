import json

import pytest

from app.domain.nrim.simulation.dataset_manifest import (
    DatasetManifest,
    ScenarioManifestRecord,
)
from app.domain.nrim.simulation.locked_test_governance import (
    create_locked_test_seal,
    locked_test_is_unopened,
    open_locked_test_once,
    verify_locked_test_seal,
)
from app.domain.nrim.simulation.scenario_design import (
    DatasetSplit,
)


def make_record(
    *,
    tmp_path,
    scenario_id: str,
    split: DatasetSplit,
    topology: str,
) -> ScenarioManifestRecord:
    observable = tmp_path / f"{scenario_id}.observable.json"
    hidden = tmp_path / f"{scenario_id}.hidden.json"
    observable.write_text(
        json.dumps({"scenario_id": scenario_id}),
        encoding="utf-8",
    )
    hidden.write_text(
        json.dumps({"scenario_id": scenario_id}),
        encoding="utf-8",
    )
    return ScenarioManifestRecord(
        scenario_id=scenario_id,
        pair_id=f"pair_{scenario_id}",
        environment_id=f"environment_{scenario_id}",
        topology_fingerprint=topology,
        split=split,
        failure_type="healthy",
        healthy=True,
        room_count=100,
        floor_count=4,
        missingness_mode="none",
        missing_fraction=0.0,
        observable_path=str(observable),
        hidden_path=str(hidden),
        topology_seed=1,
        workload_seed=2,
        observation_seed=3,
        fault_seed=4,
    )


def test_locked_test_can_open_only_once_after_acceptance(
    tmp_path,
) -> None:
    manifest = DatasetManifest(
        dataset_name="test",
        dataset_version="0.4.0",
        simulator_version="0.5.0",
        generation_seed=42,
        records=[
            make_record(
                tmp_path=tmp_path,
                scenario_id="train",
                split=DatasetSplit.TRAIN,
                topology="topology_train",
            ),
            make_record(
                tmp_path=tmp_path,
                scenario_id="locked",
                split=DatasetSplit.LOCKED_TEST,
                topology="topology_locked",
            ),
        ],
    )
    seal = create_locked_test_seal(
        manifest=manifest,
        output_dir=tmp_path / "governance",
    )
    seal_path = (
        tmp_path
        / "governance"
        / "locked_test_seal.json"
    )
    ledger_path = (
        tmp_path
        / "governance"
        / "locked_test_access.json"
    )

    assert verify_locked_test_seal(
        seal_path=seal_path
    )["commitment_sha256"] == seal[
        "commitment_sha256"
    ]
    assert locked_test_is_unopened(ledger_path)
    with pytest.raises(PermissionError):
        open_locked_test_once(
            seal_path=seal_path,
            access_ledger_path=ledger_path,
            validation_acceptance={"risk": False},
        )
    locked = open_locked_test_once(
        seal_path=seal_path,
        access_ledger_path=ledger_path,
        validation_acceptance={
            "risk": True,
            "recall": True,
        },
    )
    assert locked["records"][0]["split"] == "locked_test"
    assert not locked_test_is_unopened(ledger_path)
    with pytest.raises(PermissionError):
        open_locked_test_once(
            seal_path=seal_path,
            access_ledger_path=ledger_path,
            validation_acceptance={"risk": True},
        )

