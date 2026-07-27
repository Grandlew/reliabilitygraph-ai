from __future__ import annotations

import json
from pathlib import Path

from .candidate_evidence import CandidateSet, CausalEvidenceSet
from .decision_events import DecisionEvent
from .decision_projection import DecisionComparison, DecisionSnapshot
from .hashing import bytes_hash


DECISION_SCHEMA_MODELS = {
    "candidate_set": CandidateSet,
    "causal_evidence_set": CausalEvidenceSet,
    "decision_comparison": DecisionComparison,
    "decision_event": DecisionEvent,
    "decision_snapshot": DecisionSnapshot,
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_schema_version(version: str) -> None:
    try:
        major = int(version.split(".", 1)[0])
    except (ValueError, IndexError) as error:
        raise ValueError("Schema version must be semantic") from error
    if major != 0:
        raise ValueError("Unsupported decision-schema major version")


def export_decision_schemas(destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, model in sorted(DECISION_SCHEMA_MODELS.items()):
        path = destination / f"{name}.schema.json"
        payload = _json_bytes(model.model_json_schema())
        path.write_bytes(payload)
        paths[name] = path
        hashes[path.name] = bytes_hash(payload)
    manifest = {
        "schema_version": "0.7.2",
        "schemas": hashes,
        "compatibility": {
            "frozen_v0.6": "wrapped_without_reinterpretation",
            "unknown_major_versions": "rejected",
        },
        "truth_status": "synthetic_software_contracts_only",
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    paths["manifest"] = manifest_path
    return paths
