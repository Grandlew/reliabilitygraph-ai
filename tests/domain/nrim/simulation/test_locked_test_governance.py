import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from app.domain.nrim.simulation.dataset_manifest import (
    DatasetManifest,
    ScenarioManifestRecord,
)
from app.domain.nrim.simulation.locked_test_governance import (
    create_locked_test_seal,
    locked_test_is_unopened,
    open_locked_test_once,
    resolve_manifest_record_path,
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
    recorded_path_style: str | None = None,
) -> ScenarioManifestRecord:
    split_dir = tmp_path / split.value
    split_dir.mkdir(parents=True, exist_ok=True)
    observable = split_dir / f"{scenario_id}.observable.json"
    hidden = split_dir / f"{scenario_id}.hidden.json"
    observable.write_text(
        json.dumps({"scenario_id": scenario_id}),
        encoding="utf-8",
    )
    hidden.write_text(
        json.dumps({"scenario_id": scenario_id}),
        encoding="utf-8",
    )
    if recorded_path_style == "windows":
        recorded_root = (
            rf"C:\retired\source\day08_dataset_v06\{split.value}"
        )
        observable_path = (
            recorded_root + rf"\{scenario_id}.observable.json"
        )
        hidden_path = recorded_root + rf"\{scenario_id}.hidden.json"
    elif recorded_path_style == "posix":
        recorded_root = (
            f"/retired/source/day08_dataset_v06/{split.value}"
        )
        observable_path = (
            recorded_root + f"/{scenario_id}.observable.json"
        )
        hidden_path = recorded_root + f"/{scenario_id}.hidden.json"
    else:
        observable_path = str(observable)
        hidden_path = str(hidden)
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
        observable_path=observable_path,
        hidden_path=hidden_path,
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


@pytest.mark.parametrize(
    "recorded_path_style",
    ["windows", "posix"],
)
def test_seal_verification_ignores_stale_absolute_provenance_paths(
    tmp_path: Path,
    recorded_path_style: str,
) -> None:
    source = tmp_path / "source"
    locked_record = make_record(
        tmp_path=source,
        scenario_id="locked",
        split=DatasetSplit.LOCKED_TEST,
        topology="topology_locked",
        recorded_path_style=recorded_path_style,
    )
    manifest = DatasetManifest(
        dataset_name="test",
        dataset_version="0.4.0",
        simulator_version="0.5.0",
        generation_seed=42,
        records=[
            make_record(
                tmp_path=source,
                scenario_id="train",
                split=DatasetSplit.TRAIN,
                topology="topology_train",
                recorded_path_style=recorded_path_style,
            ),
            locked_record,
        ],
    )
    governance = source / "governance"
    create_locked_test_seal(
        manifest=manifest,
        output_dir=governance,
    )
    relocated = tmp_path / "relocated"
    source.rename(relocated)
    seal_path = (
        relocated / "governance" / "locked_test_seal.json"
    )

    assert not source.exists()
    assert not Path(locked_record.observable_path).exists()
    assert verify_locked_test_seal(
        seal_path=seal_path
    )["scenario_count"] == 1


def test_locked_scenario_checkout_policy_preserves_sealed_bytes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    dataset = Path(
        "app/domain/nrim/examples/simulation/day08_dataset_v06"
    )
    governance = root / dataset / "governance"
    seal = json.loads(
        (governance / "locked_test_seal.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (governance / "locked_test_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    governed_paths = []
    expected_by_path = {}
    for record in manifest["records"]:
        for kind in ("observable", "hidden"):
            relative_path = (
                dataset
                / str(record["split"])
                / f"{record['scenario_id']}.{kind}.json"
            ).as_posix()
            governed_paths.append(relative_path)
            expected_by_path[relative_path] = seal["file_hashes"][
                f"{record['scenario_id']}:{kind}"
            ]

    path_input = "\n".join(governed_paths) + "\n"
    attributes = subprocess.run(
        ["git", "check-attr", "--stdin", "text", "eol"],
        cwd=root,
        input=path_input.encode("utf-8"),
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8").splitlines()
    assert len(governed_paths) == 480
    assert len(attributes) == len(governed_paths) * 2
    assert all(
        line.endswith(": text: set")
        or line.endswith(": eol: crlf")
        for line in attributes
    )

    for autocrlf in ("false", "true"):
        checkout = tmp_path / f"checkout-{autocrlf}"
        checkout.mkdir()
        subprocess.run(
            [
                "git",
                "-c",
                f"core.autocrlf={autocrlf}",
                "checkout-index",
                "--force",
                f"--prefix={checkout.as_posix()}/",
                "--stdin",
            ],
            cwd=root,
            input=path_input.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        for relative_path, expected_hash in expected_by_path.items():
            checked_out = (checkout / relative_path).read_bytes()
            assert (
                hashlib.sha256(checked_out).hexdigest()
                == expected_hash
            )


@pytest.mark.parametrize(
    "filename",
    ["locked_test_manifest.json", "development_manifest.json"],
)
def test_seal_rejects_missing_colocated_manifest(
    tmp_path: Path,
    filename: str,
) -> None:
    manifest = DatasetManifest(
        dataset_name="test",
        dataset_version="0.4.0",
        simulator_version="0.5.0",
        generation_seed=42,
        records=[
            make_record(
                tmp_path=tmp_path,
                scenario_id="locked",
                split=DatasetSplit.LOCKED_TEST,
                topology="topology_locked",
            )
        ],
    )
    governance = tmp_path / "governance"
    create_locked_test_seal(
        manifest=manifest,
        output_dir=governance,
    )
    (governance / filename).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Required colocated artifact is missing",
    ):
        verify_locked_test_seal(
            seal_path=governance / "locked_test_seal.json"
        )


@pytest.mark.parametrize(
    ("recorded_path", "message"),
    [
        (
            r"C:\retired\governance\other.json",
            "wrong filename",
        ),
        (
            "../locked_test_manifest.json",
            "contains traversal",
        ),
    ],
)
def test_seal_rejects_invalid_recorded_manifest_path(
    tmp_path: Path,
    recorded_path: str,
    message: str,
) -> None:
    manifest = DatasetManifest(
        dataset_name="test",
        dataset_version="0.4.0",
        simulator_version="0.5.0",
        generation_seed=42,
        records=[
            make_record(
                tmp_path=tmp_path,
                scenario_id="locked",
                split=DatasetSplit.LOCKED_TEST,
                topology="topology_locked",
            )
        ],
    )
    governance = tmp_path / "governance"
    create_locked_test_seal(
        manifest=manifest,
        output_dir=governance,
    )
    seal_path = governance / "locked_test_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["locked_manifest_path"] = recorded_path
    seal_path.write_text(json.dumps(seal), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_locked_test_seal(seal_path=seal_path)


def test_seal_rejects_manifest_and_scenario_content_changes(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        dataset_name="test",
        dataset_version="0.4.0",
        simulator_version="0.5.0",
        generation_seed=42,
        records=[
            make_record(
                tmp_path=tmp_path,
                scenario_id="locked",
                split=DatasetSplit.LOCKED_TEST,
                topology="topology_locked",
            )
        ],
    )
    observable = tmp_path / "locked_test/locked.observable.json"
    original_observable = (
        b'{\r\n  "scenario_id": "locked"\r\n}\r\n'
    )
    observable.write_bytes(original_observable)
    governance = tmp_path / "governance"
    create_locked_test_seal(
        manifest=manifest,
        output_dir=governance,
    )
    seal_path = governance / "locked_test_seal.json"
    assert verify_locked_test_seal(
        seal_path=seal_path
    )["scenario_count"] == 1

    observable.write_bytes(
        original_observable.replace(b'"locked"', b'"changed"')
    )

    with pytest.raises(
        ValueError,
        match="Locked scenario files no longer match their seal",
    ):
        verify_locked_test_seal(seal_path=seal_path)

    observable.write_bytes(original_observable)
    locked_manifest = governance / "locked_test_manifest.json"
    payload = json.loads(locked_manifest.read_text(encoding="utf-8"))
    payload["dataset_name"] = "changed"
    locked_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Locked manifest no longer matches its seal",
    ):
        verify_locked_test_seal(seal_path=seal_path)


@pytest.mark.parametrize(
    "recorded_path",
    [
        r"C:\retired\locked_test\wrong.observable.json",
        "../locked.observable.json",
    ],
)
def test_record_resolver_rejects_wrong_filename_and_traversal(
    tmp_path: Path,
    recorded_path: str,
) -> None:
    record = make_record(
        tmp_path=tmp_path,
        scenario_id="locked",
        split=DatasetSplit.LOCKED_TEST,
        topology="topology_locked",
    ).model_dump(mode="json")
    record["observable_path"] = recorded_path

    with pytest.raises(ValueError):
        resolve_manifest_record_path(
            manifest_path=(
                tmp_path / "governance/locked_test_manifest.json"
            ),
            record=record,
            kind="observable",
        )


def test_seal_rejects_wrong_seal_filename(
    tmp_path: Path,
) -> None:
    wrong_path = tmp_path / "renamed_seal.json"
    wrong_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be named"):
        verify_locked_test_seal(seal_path=wrong_path)

