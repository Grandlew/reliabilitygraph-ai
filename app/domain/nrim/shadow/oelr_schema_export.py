from __future__ import annotations

import json
from pathlib import Path

from .evidence_anchor import (
    AnchorReceipt,
    EvidenceCheckpoint,
    SignedCheckpoint,
)
from .protocol_compiler import (
    CompiledProtocol,
    OELRProtocol,
    StatisticalContract,
)
from .readiness import (
    DressRehearsalEvidence,
    ReadOnlyBoundaryAttestation,
)
from .sampling import SamplingPlan, SamplingResult
from .semantic_acceptance import CollectorSemanticContract
from .storage_qualification import StorageQualificationEvidence
from .topology_acceptance import TopologySemanticContract


OELR_SCHEMA_MODELS = {
    "oelr_protocol": OELRProtocol,
    "compiled_protocol": CompiledProtocol,
    "statistical_contract": StatisticalContract,
    "collector_semantic_contract": CollectorSemanticContract,
    "topology_semantic_contract": TopologySemanticContract,
    "evidence_checkpoint": EvidenceCheckpoint,
    "signed_checkpoint": SignedCheckpoint,
    "anchor_receipt": AnchorReceipt,
    "sampling_plan": SamplingPlan,
    "sampling_result": SamplingResult,
    "read_only_boundary_attestation": ReadOnlyBoundaryAttestation,
    "dress_rehearsal_evidence": DressRehearsalEvidence,
    "storage_qualification_evidence": StorageQualificationEvidence,
}


def export_oelr_schemas(destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, model in sorted(OELR_SCHEMA_MODELS.items()):
        path = destination / f"{name}.schema.json"
        path.write_text(
            json.dumps(
                model.model_json_schema(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        paths[name] = path
    return paths
