from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..hashing import (
    bytes_hash,
    canonical_hash,
    canonical_json,
    file_hash,
)
from .contracts import (
    ClaimCeiling,
    DeploymentPack,
    DomainPack,
    GateDecision,
    SignatureMetadata,
    SignatureState,
    utc,
)
from .feature_reconstruction import ReconstructionState
from .qualification_measurements import QualificationMeasurements
from .replay_protocol import HistoricalReplayProtocol
from .signal_registry import SignalRegistry
from .source_inventory import (
    ReadOnlyCapabilityAttestation,
    SourceInventory,
)
from .trusted_signers import (
    SignerRole,
    TrustedSignerRegistry,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class QualificationRequest(StrictModel):
    domain_pack: DomainPack
    deployment_pack: DeploymentPack
    signal_registry: SignalRegistry
    source_inventory: SourceInventory
    capability_attestation: ReadOnlyCapabilityAttestation
    replay_protocol: HistoricalReplayProtocol
    measurements: QualificationMeasurements
    qualification_cutoff_utc: datetime

    @field_validator("qualification_cutoff_utc")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        return utc(value)


class QualificationBundle(StrictModel):
    schema_version: str = "0.8.0"
    qualification_id: str = Field(
        pattern=r"^qualification_[0-9a-f]{32}$"
    )
    decision: GateDecision
    reason_codes: tuple[str, ...]
    criteria: dict[str, bool]
    claim_ceiling: ClaimCeiling
    authorized_claim: str
    domain_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signal_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measurements_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trusted_signer_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthetic_evidence_only: bool
    model_or_threshold_tuning_authorized: bool = False
    operational_write_authorized: bool = False
    prospective_performance_claim_authorized: bool = False

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


def evaluate_qualification(
    request: QualificationRequest,
    *,
    trusted_signer_registry: TrustedSignerRegistry,
) -> QualificationBundle:
    request = QualificationRequest.model_validate(
        request.model_dump(mode="json")
    )
    trusted_signer_registry = TrustedSignerRegistry.model_validate(
        trusted_signer_registry.model_dump(mode="json")
    )
    measurements = request.measurements
    registry = trusted_signer_registry
    deployment = request.deployment_pack.deployment_pseudonym
    at_utc = request.qualification_cutoff_utc
    artifact_signatures = (
        request.domain_pack.signature,
        request.deployment_pack.signature,
        request.replay_protocol.signature,
    )
    criteria = {
        "lawful_scope": request.deployment_pack.lawful_scope_approved,
        "operator_approval": request.deployment_pack.operator_approved,
        "verified_deployment_signature": registry.verify(
            request.deployment_pack.signature,
            expected_payload_sha256=request.deployment_pack.content_hash(),
            required_role=SignerRole.DEPLOYMENT_AUTHORITY,
            deployment_pseudonym=deployment,
            at_utc=at_utc,
        ),
        "verified_domain_signature": registry.verify(
            request.domain_pack.signature,
            expected_payload_sha256=request.domain_pack.content_hash(),
            required_role=SignerRole.DOMAIN_AUTHORITY,
            deployment_pseudonym=deployment,
            at_utc=at_utc,
        ),
        "verified_replay_protocol_signature": registry.verify(
            request.replay_protocol.signature,
            expected_payload_sha256=request.replay_protocol.content_hash(),
            required_role=SignerRole.REPLAY_GOVERNANCE,
            deployment_pseudonym=deployment,
            at_utc=at_utc,
        ),
        "distinct_artifact_signers": (
            all(item.key_id is not None for item in artifact_signatures)
            and len({item.key_id for item in artifact_signatures}) == 3
        ),
        "contract_compatibility": (
            request.deployment_pack.domain_pack_sha256
            == request.domain_pack.content_hash()
            and request.domain_pack.metric_registry_sha256
            == request.signal_registry.content_hash()
            and request.source_inventory.deployment_pseudonym
            == request.deployment_pack.deployment_pseudonym
        ),
        "source_inventory_bound": (
            set(request.deployment_pack.approved_source_ids)
            == {item.source_id for item in request.source_inventory.sources}
        ),
        "read_only_capability": request.capability_attestation.passed,
        "collector_semantics": (
            measurements.semantic.mapping_fraction == 1.0
            and measurements.semantic.positive_fixture_fraction == 1.0
            and measurements.semantic.mutation_rejection_fraction == 1.0
        ),
        "topology": (
            measurements.topology.reconstruction_fraction == 1.0
            and measurements.topology.mutation_rejection_fraction == 1.0
        ),
        "feature_reconstructability": (
            measurements.feature_reconstruction.state
            is ReconstructionState.COMPLETE
            and measurements.feature_reconstruction.required_feature_fraction
            >= 0.95
            and measurements.feature_reconstruction.inference_call_count == 0
        ),
        "incident_alignment": (
            measurements.outcomes.incident_alignment_fraction >= 0.90
        ),
        "root_cause_mapping": (
            measurements.outcomes.root_cause_mapping_fraction >= 0.95
        ),
        "outcome_inventory": measurements.outcomes.inventory_complete,
        "zero_leakage": (
            measurements.topology.future_leakage_count == 0
            and measurements.privacy.identifier_leakage_count == 0
        ),
        "no_tuning": (
            measurements.tuning_audit.prohibited_event_count == 0
            and measurements.feature_reconstruction.tuning_event_count == 0
        ),
        "deterministic_tamper_evident_evidence": (
            measurements.reproducibility.determinism_fraction == 1.0
            and measurements.reproducibility.tamper_detection_fraction == 1.0
        ),
    }
    synthetic = not request.deployment_pack.real_operator_data
    reasons: list[str] = []
    if synthetic:
        decision = GateDecision.REAL_DATA_REQUIRED
        reasons.append("LAWFUL_REAL_DEPLOYMENT_PACK_REQUIRED")
    else:
        failed = [name for name, passed in criteria.items() if not passed]
        if failed:
            decision = GateDecision.BLOCKED
            reasons.extend(f"CRITERION_FAILED:{name}" for name in failed)
        else:
            decision = GateDecision.REAL_REPLAY_READY
    if decision is GateDecision.REAL_REPLAY_READY:
        ceiling = ClaimCeiling.ONE_QUALIFIED_DEPLOYMENT_REPLAY_COMPATIBILITY
        claim = (
            "Compatible with one qualified real IPTV deployment for "
            "preregistered historical replay."
        )
    else:
        ceiling = ClaimCeiling.ENGINEERING_READINESS_ONLY
        claim = (
            "Engineering qualification infrastructure only; no real IPTV "
            "deployment has passed the external gate."
        )
    identity = {
        "domain_pack": request.domain_pack.content_hash(),
        "deployment_pack": request.deployment_pack.content_hash(),
        "signal_registry": request.signal_registry.content_hash(),
        "source_inventory": request.source_inventory.content_hash(),
        "replay_protocol": request.replay_protocol.content_hash(),
        "measurements": measurements.content_hash,
        "trusted_signer_registry": registry.content_hash,
        "qualification_cutoff_utc": at_utc,
        "decision": decision.value,
        "reason_codes": sorted(reasons),
    }
    return QualificationBundle(
        qualification_id="qualification_" + canonical_hash(identity)[:32],
        decision=decision,
        reason_codes=tuple(sorted(reasons)),
        criteria=dict(sorted(criteria.items())),
        claim_ceiling=ceiling,
        authorized_claim=claim,
        domain_pack_sha256=identity["domain_pack"],
        deployment_pack_sha256=identity["deployment_pack"],
        signal_registry_sha256=identity["signal_registry"],
        source_inventory_sha256=identity["source_inventory"],
        replay_protocol_sha256=identity["replay_protocol"],
        measurements_sha256=identity["measurements"],
        trusted_signer_registry_sha256=identity[
            "trusted_signer_registry"
        ],
        synthetic_evidence_only=synthetic,
    )


_SCHEMA_MODELS = {
    "deployment_pack": DeploymentPack,
    "domain_pack": DomainPack,
    "qualification_bundle": QualificationBundle,
    "qualification_request": QualificationRequest,
    "qualification_measurements": QualificationMeasurements,
    "replay_protocol": HistoricalReplayProtocol,
    "signal_registry": SignalRegistry,
    "source_inventory": SourceInventory,
    "trusted_signer_registry": TrustedSignerRegistry,
}


def export_schemas(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, model in sorted(_SCHEMA_MODELS.items()):
        payload = (
            canonical_json(model.model_json_schema()).encode("utf-8") + b"\n"
        )
        path = directory / f"{name}.schema.json"
        path.write_bytes(payload)
        hashes[path.name] = bytes_hash(payload)
    return hashes


def seal_evidence_bundle(
    *,
    request: QualificationRequest,
    trusted_signer_registry: TrustedSignerRegistry,
    output_directory: Path,
    signature: SignatureMetadata | None = None,
    signature_provider: Callable[[str], SignatureMetadata] | None = None,
) -> dict[str, Any]:
    if signature is not None and signature_provider is not None:
        raise ValueError("Provide signature metadata or a provider, not both")
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_qualification(
        request,
        trusted_signer_registry=trusted_signer_registry,
    )
    result_path = output / "qualification.json"
    request_path = output / "request.json"
    trusted_registry_path = output / "trusted_signer_registry.json"
    result_path.write_bytes(
        canonical_json(result.model_dump(mode="json")).encode("utf-8") + b"\n"
    )
    request_path.write_bytes(
        canonical_json(request.model_dump(mode="json")).encode("utf-8") + b"\n"
    )
    trusted_registry_path.write_bytes(
        canonical_json(
            trusted_signer_registry.model_dump(mode="json")
        ).encode("utf-8")
        + b"\n"
    )
    schema_hashes = export_schemas(output / "schemas")
    files = {
        "qualification.json": file_hash(result_path),
        "request.json": file_hash(request_path),
        "trusted_signer_registry.json": file_hash(trusted_registry_path),
        **{
            f"schemas/{name}": digest
            for name, digest in sorted(schema_hashes.items())
        },
    }
    manifest_payload = {
        "schema_version": "0.8.0",
        "qualification_id": result.qualification_id,
        "decision": result.decision.value,
        "files": dict(sorted(files.items())),
        "synthetic_evidence_only": result.synthetic_evidence_only,
        "claim_ceiling": result.claim_ceiling.value,
    }
    manifest_payload_sha256 = canonical_hash(manifest_payload)
    if signature_provider is not None:
        signature = signature_provider(manifest_payload_sha256)
    if signature is None:
        raise ValueError("Qualification manifest signature metadata is required")
    if signature.state is SignatureState.VERIFIED and not (
        trusted_signer_registry.verify(
            signature,
            expected_payload_sha256=manifest_payload_sha256,
            required_role=SignerRole.QUALIFICATION_AUTHORITY,
            deployment_pseudonym=request.deployment_pack.deployment_pseudonym,
            at_utc=request.qualification_cutoff_utc,
        )
    ):
        raise ValueError("Qualification manifest signature is invalid")
    artifact_key_ids = {
        request.domain_pack.signature.key_id,
        request.deployment_pack.signature.key_id,
        request.replay_protocol.signature.key_id,
    }
    if (
        signature.state is SignatureState.VERIFIED
        and signature.key_id in artifact_key_ids
    ):
        raise ValueError(
            "Qualification manifest signer must be distinct from artifact signers"
        )
    if (
        result.decision is GateDecision.REAL_REPLAY_READY
        and signature.state is not SignatureState.VERIFIED
    ):
        raise ValueError("REAL_REPLAY_READY requires a verified manifest")
    manifest = {
        **manifest_payload,
        "manifest_payload_sha256": manifest_payload_sha256,
        "signature": signature.model_dump(mode="json"),
    }
    (output / "manifest.json").write_bytes(
        canonical_json(manifest).encode("utf-8") + b"\n"
    )
    return manifest


def verify_evidence_bundle(
    directory: Path,
    *,
    trusted_signer_registry: TrustedSignerRegistry,
) -> dict[str, Any]:
    root = directory.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "0.8.0":
        raise ValueError("Qualification bundle schema version differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Qualification bundle manifest is empty")
    for relative, expected in sorted(files.items()):
        path_value = Path(relative)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("Qualification manifest path is unsafe")
        path = (root / path_value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Qualification artifact is missing")
        if file_hash(path) != expected:
            raise ValueError(f"Qualification artifact differs: {relative}")
    result = QualificationBundle.model_validate_json(
        (root / "qualification.json").read_text(encoding="utf-8")
    )
    request = QualificationRequest.model_validate_json(
        (root / "request.json").read_text(encoding="utf-8")
    )
    bundled_registry = TrustedSignerRegistry.model_validate_json(
        (root / "trusted_signer_registry.json").read_text(encoding="utf-8")
    )
    if bundled_registry.content_hash != trusted_signer_registry.content_hash:
        raise ValueError(
            "Bundled signer registry differs from the trusted registry"
        )
    expected_result = evaluate_qualification(
        request,
        trusted_signer_registry=trusted_signer_registry,
    )
    if expected_result.content_hash != result.content_hash:
        raise ValueError("Qualification result is not derived from its evidence")
    if (
        result.synthetic_evidence_only
        and result.decision is GateDecision.REAL_REPLAY_READY
    ):
        raise ValueError("Synthetic evidence cannot pass the real-data gate")
    if manifest.get("decision") != result.decision.value:
        raise ValueError("Manifest and qualification decision differ")
    signature = SignatureMetadata.model_validate(manifest.get("signature"))
    manifest_payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"signature", "manifest_payload_sha256"}
    }
    expected_payload = canonical_hash(manifest_payload)
    if manifest.get("manifest_payload_sha256") != expected_payload:
        raise ValueError("Qualification manifest payload commitment differs")
    if signature.state is SignatureState.VERIFIED and not (
        trusted_signer_registry.verify(
            signature,
            expected_payload_sha256=expected_payload,
            required_role=SignerRole.QUALIFICATION_AUTHORITY,
            deployment_pseudonym=request.deployment_pack.deployment_pseudonym,
            at_utc=request.qualification_cutoff_utc,
        )
    ):
        raise ValueError("Qualification manifest signature is invalid")
    artifact_key_ids = {
        request.domain_pack.signature.key_id,
        request.deployment_pack.signature.key_id,
        request.replay_protocol.signature.key_id,
    }
    if (
        signature.state is SignatureState.VERIFIED
        and signature.key_id in artifact_key_ids
    ):
        raise ValueError(
            "Qualification manifest signer must be distinct from artifact signers"
        )
    if (
        result.decision is GateDecision.REAL_REPLAY_READY
        and signature.state is not SignatureState.VERIFIED
    ):
        raise ValueError("REAL_REPLAY_READY manifest is not signed")
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--request", type=Path, required=True)
    qualify.add_argument(
        "--trusted-signer-registry",
        type=Path,
        required=True,
    )
    qualify.add_argument(
        "--manifest-signature",
        type=Path,
        required=True,
    )
    qualify.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument(
        "--trusted-signer-registry",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    trusted_registry = TrustedSignerRegistry.model_validate_json(
        args.trusted_signer_registry.read_text(encoding="utf-8")
    )
    if args.command == "qualify":
        request = QualificationRequest.model_validate_json(
            args.request.read_text(encoding="utf-8")
        )
        signature = SignatureMetadata.model_validate_json(
            args.manifest_signature.read_text(encoding="utf-8")
        )
        manifest = seal_evidence_bundle(
            request=request,
            trusted_signer_registry=trusted_registry,
            output_directory=args.output_dir,
            signature=signature,
        )
    else:
        manifest = verify_evidence_bundle(
            args.directory,
            trusted_signer_registry=trusted_registry,
        )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
