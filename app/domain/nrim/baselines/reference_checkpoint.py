from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .incident_detector import IncidentDetectorModel
from .learned_fusion import RootCauseFusionModel


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reference_checkpoint(
    *,
    incident_model: IncidentDetectorModel,
    fusion_model: RootCauseFusionModel,
    incident_threshold: float,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    payload = {
        "checkpoint_version": "1.0.0",
        "dataset_fingerprint": dataset_fingerprint,
        "incident_threshold": incident_threshold,
        "incident_model": asdict(incident_model),
        "learned_fusion_model": asdict(fusion_model),
        "ranking_policy": {
            "stage2_reference": "learned_fusion",
            "topology_retained": fusion_model.use_topology,
            "modified_by_temporal_gate": False,
        },
    }
    payload["checkpoint_sha256"] = _canonical_hash(payload)
    return payload


def save_reference_checkpoint(
    *,
    checkpoint: dict[str, Any],
    path: Path,
) -> None:
    expected_hash = str(checkpoint["checkpoint_sha256"])
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256")
    if _canonical_hash(unsigned) != expected_hash:
        raise ValueError("Reference checkpoint hash is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def load_reference_checkpoint(
    *,
    path: Path,
    expected_dataset_fingerprint: str | None = None,
) -> tuple[IncidentDetectorModel, RootCauseFusionModel, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(payload.pop("checkpoint_sha256"))
    if _canonical_hash(payload) != expected_hash:
        raise ValueError("Reference checkpoint hash is invalid")
    if (
        expected_dataset_fingerprint is not None
        and payload["dataset_fingerprint"]
        != expected_dataset_fingerprint
    ):
        raise ValueError("Reference checkpoint dataset fingerprint differs")
    incident = dict(payload["incident_model"])
    fusion = dict(payload["learned_fusion_model"])
    for field in ("feature_names", "means", "scales", "weights"):
        incident[field] = tuple(incident[field])
        fusion[field] = tuple(fusion[field])
    return (
        IncidentDetectorModel(**incident),
        RootCauseFusionModel(**fusion),
        float(payload["incident_threshold"]),
    )

