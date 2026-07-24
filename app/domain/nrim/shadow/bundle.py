from __future__ import annotations

import base64
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from app.domain.nrim.baselines.dual_path_episode_gate import (
    DualPathConfig,
    HealthyResidualModel,
    confirm_observable_impact,
)
from app.domain.nrim.baselines.incident_detector import (
    IncidentDetectorModel,
)
from app.domain.nrim.baselines.learned_fusion import (
    RootCauseFusionModel,
    learned_fusion_scores,
)
from app.domain.nrim.baselines.shift_sentinels import (
    DomainDiscriminator,
    FeatureSupport,
    ShiftSentinelModel,
    assess_support,
)
from app.domain.nrim.simulation.feature_schema import FeatureSchema

from .hashing import bytes_hash, canonical_hash, canonical_json, file_hash


BUNDLE_FORMAT_VERSION = "1.0.0"
REQUIRED_ARTIFACTS = {
    "models.json",
    "policy.json",
    "feature_schema.json",
    "categories.json",
    "golden_snapshots.json",
    "provenance.json",
}


class BundleIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenInferenceBundle:
    incident_model: IncidentDetectorModel
    residual_model: HealthyResidualModel
    support_model: ShiftSentinelModel
    fusion_model: RootCauseFusionModel
    policy: DualPathConfig
    incident_threshold: float
    feature_schema: FeatureSchema
    bundle_hash: str
    policy_hash: str
    provenance: dict[str, Any]
    golden_snapshots: tuple[dict[str, Any], ...]

    def evaluate_window(
        self,
        *,
        window: dict[str, Any],
        metadata: dict[str, Any],
        prior_logits: Sequence[float] = (),
    ) -> dict[str, Any]:
        probability = self.incident_model.predict_score(window)
        residual = self.residual_model.standardized_residual(
            window=window,
            probability=probability,
            metadata=metadata,
            prior_logits=prior_logits,
        )
        support = assess_support(
            model=self.support_model,
            window=window,
            metadata=metadata,
        )
        impact = confirm_observable_impact(
            window=window,
            score_threshold=self.policy.impact_score_threshold,
            minimum_families=self.policy.impact_minimum_families,
            minimum_node_fraction=(
                self.policy.impact_minimum_node_fraction
            ),
            require_causal_consistency=(
                self.policy.require_causal_consistency
            ),
        )
        ranking = sorted(
            learned_fusion_scores(
                window=window,
                model=self.fusion_model,
            ),
            key=lambda item: (-item.score, item.node_id),
        )
        return {
            "stage1_probability": probability,
            "healthy_residual": residual,
            "support": {
                "status": support.status.value,
                "score": support.support_score,
                "axes": dict(support.axis_scores),
            },
            "impact": {
                "confirmed": impact.confirmed,
                "score": impact.score,
                "family_count": impact.family_count,
                "affected_node_fraction": impact.affected_node_fraction,
                "causal_consistent": impact.causal_consistent,
                "families": list(impact.families),
            },
            "stage2": [
                {
                    "component": item.node_id,
                    "score": item.score,
                    "evidence_hash": canonical_hash(item.evidence),
                }
                for item in ranking
            ],
        }

    def verify_golden_snapshots(
        self,
        *,
        numeric_tolerance: float = 1e-12,
    ) -> None:
        for case in self.golden_snapshots:
            actual = self.evaluate_window(
                window=case["window"],
                metadata=case["metadata"],
                prior_logits=case.get("prior_logits", ()),
            )
            _assert_equivalent(
                expected=case["expected"],
                actual=actual,
                tolerance=numeric_tolerance,
                path=case["case_id"],
            )


def _assert_equivalent(
    *,
    expected: Any,
    actual: Any,
    tolerance: float,
    path: str,
) -> None:
    if isinstance(expected, bool) or expected is None:
        if actual != expected:
            raise BundleIntegrityError(
                f"Golden mismatch at {path}: {expected!r} != {actual!r}"
            )
        return
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(expected),
            float(actual),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise BundleIntegrityError(
                f"Golden numeric mismatch at {path}: "
                f"{expected!r} != {actual!r}"
            )
        return
    if isinstance(expected, str):
        if actual != expected:
            raise BundleIntegrityError(
                f"Golden mismatch at {path}: {expected!r} != {actual!r}"
            )
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise BundleIntegrityError(f"Golden mapping mismatch at {path}")
        for key in expected:
            _assert_equivalent(
                expected=expected[key],
                actual=actual[key],
                tolerance=tolerance,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected,
        (str, bytes, bytearray),
    ):
        if not isinstance(actual, Sequence) or len(expected) != len(actual):
            raise BundleIntegrityError(f"Golden sequence mismatch at {path}")
        for index, (left, right) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _assert_equivalent(
                expected=left,
                actual=right,
                tolerance=tolerance,
                path=f"{path}[{index}]",
            )
        return
    if expected != actual:
        raise BundleIntegrityError(
            f"Golden mismatch at {path}: {expected!r} != {actual!r}"
        )


def generate_signing_keypair() -> tuple[Ed25519PrivateKey, bytes]:
    private_key = Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_pem


def create_signed_bundle(
    *,
    output_path: Path,
    public_key_path: Path,
    private_key: Ed25519PrivateKey,
    public_key_pem: bytes,
    artifacts: Mapping[str, Any],
    source_root: Path,
    source_paths: Sequence[Path],
    provenance: dict[str, Any],
) -> str:
    if set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError(
            "Bundle artifacts differ from required contract: "
            + repr(sorted(set(artifacts) ^ REQUIRED_ARTIFACTS))
        )
    artifact_bytes = {
        name: canonical_json(value).encode("utf-8")
        for name, value in artifacts.items()
    }
    source_hashes = {}
    root = source_root.resolve()
    for path in source_paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Source path lies outside source root") from error
        source_hashes[relative] = file_hash(resolved)
    public_key_fingerprint = bytes_hash(public_key_pem)
    signed_payload = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "bundle_version": "0.7.0-frozen-v0.6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_hashes": {
            name: bytes_hash(value)
            for name, value in sorted(artifact_bytes.items())
        },
        "source_hashes": dict(sorted(source_hashes.items())),
        "public_key_sha256": public_key_fingerprint,
        "provenance_sha256": canonical_hash(provenance),
    }
    signed_bytes = canonical_json(signed_payload).encode("utf-8")
    manifest = {
        "signed_payload": signed_payload,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(
            private_key.sign(signed_bytes)
        ).decode("ascii"),
    }
    bundle_hash = canonical_hash(signed_payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.write_bytes(public_key_pem)
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("manifest.json", canonical_json(manifest))
        for name in sorted(artifact_bytes):
            archive.writestr(name, artifact_bytes[name])
    return bundle_hash


def _safe_archive_names(archive: zipfile.ZipFile) -> set[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise BundleIntegrityError("Bundle contains duplicate archive entries")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise BundleIntegrityError("Bundle contains unsafe archive path")
    return set(names)


def _load_public_key(public_key_path: Path) -> Ed25519PublicKey:
    value = serialization.load_pem_public_key(public_key_path.read_bytes())
    if not isinstance(value, Ed25519PublicKey):
        raise BundleIntegrityError("Bundle trust anchor is not Ed25519")
    return value


def _tuple_fields(value: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    result = dict(value)
    for field in fields:
        result[field] = tuple(result[field])
    return result


def _models(payload: dict[str, Any]) -> tuple[
    IncidentDetectorModel,
    HealthyResidualModel,
    ShiftSentinelModel,
    RootCauseFusionModel,
]:
    incident = IncidentDetectorModel(
        **_tuple_fields(
            payload["incident_model"],
            ("feature_names", "means", "scales", "weights"),
        )
    )
    residual = HealthyResidualModel(
        **_tuple_fields(
            payload["healthy_residual_model"],
            ("feature_names", "means", "scales", "weights"),
        )
    )
    sentinel_payload = payload["support_sentinel"]
    discriminator = DomainDiscriminator(
        **_tuple_fields(
            sentinel_payload["domain_discriminator"],
            ("feature_names", "means", "scales", "weights"),
        )
    )
    sentinel = ShiftSentinelModel(
        supports=tuple(
            FeatureSupport(**item)
            for item in sentinel_payload["supports"]
        ),
        domain_discriminator=discriminator,
        support_threshold=sentinel_payload["support_threshold"],
    )
    fusion = RootCauseFusionModel(
        **_tuple_fields(
            payload["learned_fusion_model"],
            ("feature_names", "means", "scales", "weights"),
        )
    )
    return incident, residual, sentinel, fusion


def load_signed_bundle(
    *,
    bundle_path: Path,
    public_key_path: Path,
    source_root: Path,
    expected_bundle_hash: str | None = None,
    run_golden_checks: bool = True,
) -> FrozenInferenceBundle:
    public_key_bytes = public_key_path.read_bytes()
    public_key = _load_public_key(public_key_path)
    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = _safe_archive_names(archive)
        expected_names = {"manifest.json", *REQUIRED_ARTIFACTS}
        if names != expected_names:
            raise BundleIntegrityError(
                "Bundle entry set differs from registered contract"
            )
        manifest = json.loads(archive.read("manifest.json"))
        signed_payload = manifest["signed_payload"]
        if manifest.get("signature_algorithm") != "ed25519":
            raise BundleIntegrityError("Unsupported signature algorithm")
        if (
            signed_payload.get("public_key_sha256")
            != bytes_hash(public_key_bytes)
        ):
            raise BundleIntegrityError("Public-key trust anchor differs")
        try:
            public_key.verify(
                base64.b64decode(
                    manifest["signature_base64"],
                    validate=True,
                ),
                canonical_json(signed_payload).encode("utf-8"),
            )
        except (InvalidSignature, ValueError) as error:
            raise BundleIntegrityError("Bundle signature is invalid") from error
        bundle_hash = canonical_hash(signed_payload)
        if (
            expected_bundle_hash is not None
            and bundle_hash != expected_bundle_hash
        ):
            raise BundleIntegrityError("Bundle commitment differs")
        artifacts = {}
        for name in REQUIRED_ARTIFACTS:
            raw = archive.read(name)
            if (
                bytes_hash(raw)
                != signed_payload["artifact_hashes"].get(name)
            ):
                raise BundleIntegrityError(
                    f"Bundle artifact hash differs: {name}"
                )
            artifacts[name] = json.loads(raw)

    root = source_root.resolve()
    for relative, expected_hash in signed_payload[
        "source_hashes"
    ].items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise BundleIntegrityError("Unsafe registered source path") from error
        if not path.is_file() or file_hash(path) != expected_hash:
            raise BundleIntegrityError(
                f"Frozen inference source differs: {relative}"
            )
    provenance = artifacts["provenance.json"]
    if canonical_hash(provenance) != signed_payload["provenance_sha256"]:
        raise BundleIntegrityError("Bundle provenance differs")
    incident, residual, sentinel, fusion = _models(
        artifacts["models.json"]
    )
    policy_payload = artifacts["policy.json"]
    bundle = FrozenInferenceBundle(
        incident_model=incident,
        residual_model=residual,
        support_model=sentinel,
        fusion_model=fusion,
        policy=DualPathConfig(**policy_payload["dual_path_policy"]),
        incident_threshold=float(policy_payload["incident_threshold"]),
        feature_schema=FeatureSchema.model_validate(
            artifacts["feature_schema.json"]
        ),
        bundle_hash=bundle_hash,
        policy_hash=canonical_hash(policy_payload),
        provenance=provenance,
        golden_snapshots=tuple(artifacts["golden_snapshots.json"]),
    )
    if run_golden_checks:
        bundle.verify_golden_snapshots(
            numeric_tolerance=float(
                policy_payload["golden_numeric_tolerance"]
            )
        )
    return bundle


def serialize_models(
    *,
    incident_model: IncidentDetectorModel,
    residual_model: HealthyResidualModel,
    support_model: ShiftSentinelModel,
    fusion_model: RootCauseFusionModel,
) -> dict[str, Any]:
    return {
        "incident_model": asdict(incident_model),
        "healthy_residual_model": asdict(residual_model),
        "support_sentinel": asdict(support_model),
        "learned_fusion_model": asdict(fusion_model),
    }
