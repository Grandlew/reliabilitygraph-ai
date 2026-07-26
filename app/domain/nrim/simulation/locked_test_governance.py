from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset_manifest import (
    DatasetManifest,
    save_manifest,
)
from .scenario_design import DatasetSplit


_SEAL_FILENAME = "locked_test_seal.json"
_MANIFEST_FILENAMES = {
    "locked_manifest_path": "locked_test_manifest.json",
    "development_manifest_path": "development_manifest.json",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_recorded_filename(
    raw_path: Any,
    *,
    expected_filename: str,
) -> None:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Recorded artifact path must be a non-empty string")
    portable = PurePosixPath(raw_path.replace("\\", "/"))
    if ".." in portable.parts:
        raise ValueError("Recorded artifact path contains traversal")
    if portable.name != expected_filename:
        raise ValueError(
            "Recorded artifact path has the wrong filename: "
            f"expected {expected_filename!r}"
        )


def _require_colocated_file(
    *,
    directory: Path,
    filename: str,
) -> Path:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required colocated artifact is missing: {path}"
        )
    return path


def resolve_seal_manifest_path(
    *,
    seal_path: Path,
    seal: dict[str, Any],
    field: str,
) -> Path:
    """Resolve a manifest beside its seal without trusting provenance paths."""

    if seal_path.name != _SEAL_FILENAME:
        raise ValueError(
            f"Locked-test seal must be named {_SEAL_FILENAME!r}"
        )
    try:
        expected_filename = _MANIFEST_FILENAMES[field]
        raw_path = seal[field]
    except KeyError as error:
        raise ValueError(
            f"Locked-test seal is missing {field!r}"
        ) from error
    _validate_recorded_filename(
        raw_path,
        expected_filename=expected_filename,
    )
    return _require_colocated_file(
        directory=seal_path.parent,
        filename=expected_filename,
    )


def resolve_manifest_record_path(
    *,
    manifest_path: Path,
    record: dict[str, Any],
    kind: str,
) -> Path:
    """Resolve a scenario artifact from the current manifest location."""

    if kind not in {"observable", "hidden"}:
        raise ValueError(f"Unsupported scenario artifact kind: {kind!r}")
    scenario_id = str(record["scenario_id"])
    split = str(record["split"])
    try:
        DatasetSplit(split)
    except ValueError as error:
        raise ValueError(
            f"Scenario record has an invalid split: {split!r}"
        ) from error
    expected_filename = f"{scenario_id}.{kind}.json"
    _validate_recorded_filename(
        record[f"{kind}_path"],
        expected_filename=expected_filename,
    )
    manifest_directory = manifest_path.parent
    dataset_root = (
        manifest_directory.parent
        if manifest_directory.name == "governance"
        else manifest_directory
    )
    return _require_colocated_file(
        directory=dataset_root / split,
        filename=expected_filename,
    )


def create_locked_test_seal(
    *,
    manifest: DatasetManifest,
    output_dir: Path,
) -> dict[str, Any]:
    """Seal locked records before any model calibration occurs."""

    locked_records = [
        record
        for record in manifest.records
        if record.split is DatasetSplit.LOCKED_TEST
    ]
    if not locked_records:
        raise ValueError(
            "A locked-test seal requires locked records"
        )
    development_records = [
        record
        for record in manifest.records
        if record.split is not DatasetSplit.LOCKED_TEST
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    locked_manifest = manifest.model_copy(
        update={"records": locked_records}
    )
    development_manifest = manifest.model_copy(
        update={"records": development_records}
    )
    locked_manifest_path = (
        output_dir / "locked_test_manifest.json"
    )
    development_manifest_path = (
        output_dir / "development_manifest.json"
    )
    save_manifest(
        manifest=locked_manifest,
        path=locked_manifest_path,
    )
    save_manifest(
        manifest=development_manifest,
        path=development_manifest_path,
    )

    file_hashes = {}
    for record in locked_records:
        record_payload = record.model_dump(mode="json")
        for kind in ("observable", "hidden"):
            path = resolve_manifest_record_path(
                manifest_path=locked_manifest_path,
                record=record_payload,
                kind=kind,
            )
            file_hashes[
                f"{record.scenario_id}:{kind}"
            ] = file_sha256(path)
    manifest_payload = locked_manifest.model_dump(
        mode="json"
    )
    commitment_payload = {
        "locked_manifest_sha256": _sha256_bytes(
            _canonical_bytes(manifest_payload)
        ),
        "file_hashes": dict(sorted(file_hashes.items())),
    }
    seal = {
        "seal_version": "1.0.0",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "locked_split": DatasetSplit.LOCKED_TEST.value,
        "scenario_count": len(locked_records),
        "locked_manifest_path": str(
            locked_manifest_path
        ),
        "development_manifest_path": str(
            development_manifest_path
        ),
        **commitment_payload,
        "commitment_sha256": _sha256_bytes(
            _canonical_bytes(commitment_payload)
        ),
        "opened": False,
    }
    seal_path = output_dir / "locked_test_seal.json"
    seal_path.write_text(
        json.dumps(seal, indent=2),
        encoding="utf-8",
    )
    return seal


def verify_locked_test_seal(
    *,
    seal_path: Path,
) -> dict[str, Any]:
    if seal_path.name != _SEAL_FILENAME:
        raise ValueError(
            f"Locked-test seal must be named {_SEAL_FILENAME!r}"
        )
    seal = json.loads(
        seal_path.read_text(encoding="utf-8")
    )
    locked_manifest_path = resolve_seal_manifest_path(
        seal_path=seal_path,
        seal=seal,
        field="locked_manifest_path",
    )
    resolve_seal_manifest_path(
        seal_path=seal_path,
        seal=seal,
        field="development_manifest_path",
    )
    locked_payload = json.loads(
        locked_manifest_path.read_text(
            encoding="utf-8"
        )
    )
    manifest_hash = _sha256_bytes(
        _canonical_bytes(locked_payload)
    )
    if manifest_hash != seal[
        "locked_manifest_sha256"
    ]:
        raise ValueError(
            "Locked manifest no longer matches its seal"
        )
    records = locked_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Locked manifest has no records list")
    if (
        seal.get("locked_split")
        != DatasetSplit.LOCKED_TEST.value
        or len(records) != seal.get("scenario_count")
        or any(
            record.get("split")
            != DatasetSplit.LOCKED_TEST.value
            for record in records
        )
    ):
        raise ValueError(
            "Locked-test split or scenario-count invariant is invalid"
        )
    current_hashes = {}
    for record in records:
        for kind in ("observable", "hidden"):
            current_hashes[
                f"{record['scenario_id']}:{kind}"
            ] = file_sha256(
                resolve_manifest_record_path(
                    manifest_path=locked_manifest_path,
                    record=record,
                    kind=kind,
                )
            )
    expected_hashes = dict(seal["file_hashes"])
    if current_hashes != expected_hashes:
        raise ValueError(
            "Locked scenario files no longer match their seal"
        )
    commitment_payload = {
        "locked_manifest_sha256": manifest_hash,
        "file_hashes": dict(
            sorted(current_hashes.items())
        ),
    }
    if (
        _sha256_bytes(
            _canonical_bytes(commitment_payload)
        )
        != seal["commitment_sha256"]
    ):
        raise ValueError(
            "Locked-test commitment hash is invalid"
        )
    return seal


def open_locked_test_once(
    *,
    seal_path: Path,
    access_ledger_path: Path,
    validation_acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Open once, and only after every registered validation criterion."""

    failed = [
        name
        for name, passed in validation_acceptance.items()
        if not bool(passed)
    ]
    if failed:
        raise PermissionError(
            "Locked test remains sealed because validation failed: "
            + ", ".join(sorted(failed))
        )
    if access_ledger_path.exists():
        raise PermissionError(
            "Locked test has already been opened"
        )
    seal = verify_locked_test_seal(
        seal_path=seal_path
    )
    locked_manifest_path = resolve_seal_manifest_path(
        seal_path=seal_path,
        seal=seal,
        field="locked_manifest_path",
    )
    ledger = {
        "accessed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "commitment_sha256": seal[
            "commitment_sha256"
        ],
        "validation_acceptance": (
            validation_acceptance
        ),
        "purpose": (
            "single final v0.6 locked-test evaluation"
        ),
    }
    access_ledger_path.write_text(
        json.dumps(ledger, indent=2),
        encoding="utf-8",
    )
    return json.loads(
        locked_manifest_path.read_text(encoding="utf-8")
    )


def locked_test_is_unopened(
    access_ledger_path: Path,
) -> bool:
    return not access_ledger_path.exists()

