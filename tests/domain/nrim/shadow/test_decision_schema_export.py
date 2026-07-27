from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.nrim.shadow.decision_schema_export import (
    DECISION_SCHEMA_MODELS,
    export_decision_schemas,
    validate_schema_version,
)


def _bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_schema_export_is_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_decision_schemas(first)
    export_decision_schemas(second)
    assert _bytes(first) == _bytes(second)


def test_schema_manifest_hashes_every_schema(tmp_path) -> None:
    destination = tmp_path / "schemas"
    export_decision_schemas(destination)
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["schemas"]) == {
        f"{name}.schema.json" for name in DECISION_SCHEMA_MODELS
    }
    assert all(len(value) == 64 for value in manifest["schemas"].values())


@pytest.mark.parametrize("version", ["1.0.0", "2.1.0", "not-semver"])
def test_unknown_or_invalid_schema_major_is_rejected(version: str) -> None:
    with pytest.raises(ValueError):
        validate_schema_version(version)


@pytest.mark.parametrize("version", ["0.7.2", "0.8.0"])
def test_zero_major_schema_versions_are_supported(version: str) -> None:
    validate_schema_version(version)


def test_checked_in_schemas_match_exporter(tmp_path) -> None:
    generated = tmp_path / "schemas"
    export_decision_schemas(generated)
    root = Path(__file__).resolve().parents[4]
    checked_in = (
        root
        / "app/domain/nrim/examples/shadow/decision_v0_7_2/schemas"
    )
    assert _bytes(generated) == _bytes(checked_in)


def test_schema_manifest_states_compatibility_and_nonclaim(tmp_path) -> None:
    destination = tmp_path / "schemas"
    export_decision_schemas(destination)
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["compatibility"]["frozen_v0.6"] == (
        "wrapped_without_reinterpretation"
    )
    assert manifest["truth_status"] == "synthetic_software_contracts_only"
