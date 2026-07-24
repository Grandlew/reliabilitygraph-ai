from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .hashing import bytes_hash, canonical_hash, canonical_json


class FrozenPilotProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = "0.7.0"
    protocol_name: str = "NRIM v0.7 Prospective Shadow Evidence"
    registered_at_utc: datetime
    planned_start_utc: datetime
    planned_end_utc: datetime
    deployment_pseudonyms: tuple[str, ...] = Field(min_length=5)
    topology_family_count: int = Field(ge=5)
    model_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_public_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypotheses: tuple[str, ...]
    metric_definitions: tuple[str, ...]
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    path_retention_rules: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    minimum_healthy_deployment_hours: float = Field(default=3000.0, ge=3000.0)
    minimum_adjudicated_incidents: int = Field(default=30, ge=30)
    minimum_silent_weeks: float = Field(default=8.0, ge=8.0)
    no_pilot_tuning: bool = True
    predictions_hidden_from_frontline: bool = True
    automatic_remediation_enabled: bool = False

    @field_validator(
        "registered_at_utc",
        "planned_start_utc",
        "planned_end_utc",
    )
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Protocol timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if self.registered_at_utc > self.planned_start_utc:
            raise ValueError("Protocol must be registered before pilot start")
        duration = self.planned_end_utc - self.planned_start_utc
        if duration.total_seconds() < 8.0 * 7.0 * 24.0 * 3600.0:
            raise ValueError("Silent pilot must be planned for at least 8 weeks")
        if len(set(self.deployment_pseudonyms)) != len(
            self.deployment_pseudonyms
        ):
            raise ValueError("Pilot deployment pseudonyms must be unique")
        if not self.no_pilot_tuning:
            raise ValueError("v0.7 protocol must prohibit pilot tuning")
        if not self.predictions_hidden_from_frontline:
            raise ValueError("Silent pilot predictions must remain hidden")
        if self.automatic_remediation_enabled:
            raise ValueError("v0.7 cannot enable automatic remediation")
        return self


def default_protocol(
    *,
    planned_start_utc: datetime,
    planned_end_utc: datetime,
    deployment_pseudonyms: tuple[str, ...],
    topology_family_count: int,
    model_bundle_hash: str,
    bundle_public_key_sha256: str,
    feature_schema_hash: str,
    policy_hash: str,
) -> FrozenPilotProtocol:
    """Build the legacy silent-only protocol.

    New registrations should use ``protocol_compiler.OELRProtocol``. This
    compatibility builder intentionally excludes human-utility estimands,
    because operators cannot provide utility outcomes while predictions are
    hidden.
    """

    return FrozenPilotProtocol(
        registered_at_utc=datetime.now(timezone.utc),
        planned_start_utc=planned_start_utc,
        planned_end_utc=planned_end_utc,
        deployment_pseudonyms=deployment_pseudonyms,
        topology_family_count=topology_family_count,
        model_bundle_hash=model_bundle_hash,
        bundle_public_key_sha256=bundle_public_key_sha256,
        feature_schema_hash=feature_schema_hash,
        policy_hash=policy_hash,
        hypotheses=(
            "Frozen dual-path Stage 1 reaches in-support episode recall >=0.80.",
            "Non-actionable burden is <=0.10 episodes per 100 healthy hours.",
            "Frozen Stage 2 reaches MRR >=0.70 and Hits@3 >=0.85.",
            "Unsupported inputs always route to UNKNOWN or ESCALATE.",
            "Fast and slow paths are retained only if independently useful.",
        ),
        metric_definitions=(
            "episode_recall_by_deployment_and_adjudicated_incident",
            "severe_incident_recall",
            "detection_delay_distribution",
            "mean_extra_episode_fragments",
            "nonactionable_episodes_per_100_healthy_deployment_hours",
            "mrr_hits1_hits3_and_ranking_coverage",
            "unsupported_and_data_quality_blocked_rates",
        ),
        inclusion_criteria=(
            "all_confirmed_customer_impacting_incidents",
            "all_existing_monitor_incidents_including_nrim_misses",
            "all_nrim_incident_and_escalate_episodes",
            "stratified_unknown_healthy_and_random_healthy_hours",
            "all_data_quality_failures_and_topology_changes",
        ),
        exclusion_criteria=(
            "predictions_not_generated_by_registered_bundle",
            "events_without_reproducible_event_time_snapshot",
            "training_or_threshold_tuning_activity",
            "duplicate_adjudication_versions",
        ),
        path_retention_rules=(
            "fast: >=1 interval median gain OR severe rescue OR >=0.05 "
            "short recall gain, with <=0.02 excess burden",
            "slow: >=0.05 additional incidents OR weak-impact rescue OR "
            "earlier gradual warning within burden limits",
            "remove any path without independent value",
        ),
        stop_conditions=(
            "unsupported_input_emitted_healthy_or_given_stage2",
            "severe_incident_missed_due_to_reproducible_pipeline_failure",
            "future_ticket_resolution_or_identity_enters_tensor",
            "prediction_not_reproducible_from_evidence",
            "unregistered_model_or_threshold_tuning",
            "single_deployment_dominates_apparent_performance",
            "alert_burden_exceeds_existing_monitoring_without_benefit",
            "sensitive_identity_found_in_inference_or_evidence",
        ),
    )


def register_protocol(
    *,
    protocol: FrozenPilotProtocol,
    registration_path: Path,
    private_key: Ed25519PrivateKey,
    public_key_pem: bytes,
) -> str:
    """Exclusively create an immutable preregistration commitment."""

    protocol_payload = protocol.model_dump(mode="json")
    protocol_hash = canonical_hash(protocol_payload)
    signature = private_key.sign(
        canonical_json(protocol_payload).encode("utf-8")
    )
    registration = {
        "protocol": protocol_payload,
        "protocol_sha256": protocol_hash,
        "signature_algorithm": "ed25519",
        "public_key_sha256": bytes_hash(public_key_pem),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(registration_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(registration))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return protocol_hash


def verify_registered_protocol(
    *,
    registration_path: Path,
    public_key_path: Path,
) -> FrozenPilotProtocol:
    registration = json.loads(
        registration_path.read_text(encoding="utf-8")
    )
    public_pem = public_key_path.read_bytes()
    if registration["public_key_sha256"] != bytes_hash(public_pem):
        raise ValueError("Protocol trust anchor differs")
    payload = registration["protocol"]
    if registration["protocol_sha256"] != canonical_hash(payload):
        raise ValueError("Protocol commitment differs")
    public = serialization.load_pem_public_key(public_pem)
    if not isinstance(public, Ed25519PublicKey):
        raise ValueError("Protocol trust anchor must be Ed25519")
    try:
        public.verify(
            base64.b64decode(
                registration["signature_base64"],
                validate=True,
            ),
            canonical_json(payload).encode("utf-8"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("Protocol signature is invalid") from error
    return FrozenPilotProtocol.model_validate(payload)


def sign_release_report(
    *,
    report: dict[str, Any],
    private_key: Ed25519PrivateKey,
    public_key_pem: bytes,
) -> dict[str, Any]:
    """Sign a closed-pilot report; the caller persists it append-only."""

    payload_hash = canonical_hash(report)
    signature = private_key.sign(
        canonical_json(report).encode("utf-8")
    )
    return {
        "report": report,
        "report_sha256": payload_hash,
        "signature_algorithm": "ed25519",
        "public_key_sha256": bytes_hash(public_key_pem),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
