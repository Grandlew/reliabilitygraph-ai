from __future__ import annotations

import base64
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..hashing import canonical_hash
from ..privacy import assert_no_direct_identifiers


SCHEMA_VERSION = "0.8.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return value.astimezone(timezone.utc)


class GateDecision(str, Enum):
    REAL_DATA_REQUIRED = "REAL_DATA_REQUIRED"
    BLOCKED = "BLOCKED"
    REAL_REPLAY_READY = "REAL_REPLAY_READY"


class ClaimCeiling(str, Enum):
    ENGINEERING_READINESS_ONLY = "engineering_readiness_only"
    ONE_QUALIFIED_DEPLOYMENT_REPLAY_COMPATIBILITY = (
        "one_qualified_deployment_replay_compatibility"
    )


class OperatorTask(str, Enum):
    INCIDENT_CONFIRMATION = "incident_confirmation"
    BLAST_RADIUS = "blast_radius_understanding"
    HYPOTHESIS_RANKING = "hypothesis_ranking"
    EVIDENCE_INSPECTION = "evidence_inspection"
    OVERRIDE_ADJUDICATION = "override_adjudication"


class LabelKind(str, Enum):
    FAULT_PRESENCE = "fault_presence"
    OBSERVABLE_DEGRADATION = "observable_degradation"
    CUSTOMER_IMPACT = "customer_impact"


class DeploymentEnvironment(str, Enum):
    LAB = "lab"
    PREPRODUCTION = "preproduction"
    PRODUCTION_HISTORICAL_EXPORT = "production_historical_export"


class CollectionMode(str, Enum):
    OFFLINE_JSONL = "offline_jsonl"
    OFFLINE_CSV = "offline_csv"


class SignatureState(str, Enum):
    NOT_SIGNED_SYNTHETIC = "not_signed_synthetic"
    EXTERNAL_SIGNATURE_REQUIRED = "external_signature_required"
    VERIFIED = "verified"


class SignatureMetadata(StrictModel):
    state: SignatureState
    algorithm: Literal["none", "ed25519", "external"]
    key_id: str | None = Field(default=None, min_length=3, max_length=128)
    signature_base64: str | None = Field(
        default=None,
        min_length=16,
        max_length=1024,
    )
    signed_payload_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )

    @model_validator(mode="after")
    def validate_signature_state(self) -> Self:
        signed = self.state is SignatureState.VERIFIED
        fields_present = all(
            (
                self.key_id,
                self.signature_base64,
                self.signed_payload_sha256,
            )
        )
        if signed and (self.algorithm == "none" or not fields_present):
            raise ValueError("Verified signatures require complete metadata")
        if self.state is SignatureState.NOT_SIGNED_SYNTHETIC and (
            self.algorithm != "none"
            or self.key_id is not None
            or self.signature_base64 is not None
        ):
            raise ValueError("Synthetic unsigned artifacts cannot name a key")
        return self


def create_ed25519_signature(
    *,
    payload_sha256: str,
    private_key_bytes: bytes,
    key_id: str,
) -> SignatureMetadata:
    if len(private_key_bytes) != 32:
        raise ValueError("Ed25519 private key material must be 32 bytes")
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    signature = private_key.sign(payload_sha256.encode("ascii"))
    return SignatureMetadata(
        state=SignatureState.VERIFIED,
        algorithm="ed25519",
        key_id=key_id,
        signature_base64=base64.b64encode(signature).decode("ascii"),
        signed_payload_sha256=payload_sha256,
    )


def ed25519_public_key_base64(private_key_bytes: bytes) -> str:
    if len(private_key_bytes) != 32:
        raise ValueError("Ed25519 private key material must be 32 bytes")
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return base64.b64encode(
        private_key.public_key().public_bytes_raw()
    ).decode("ascii")


def verify_signature(
    signature: SignatureMetadata,
    *,
    expected_payload_sha256: str,
    trusted_public_key_base64: str | None = None,
) -> bool:
    if (
        signature.state is not SignatureState.VERIFIED
        or signature.algorithm != "ed25519"
        or signature.signed_payload_sha256 != expected_payload_sha256
        or trusted_public_key_base64 is None
        or signature.signature_base64 is None
    ):
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(trusted_public_key_base64, validate=True)
        )
        public_key.verify(
            base64.b64decode(signature.signature_base64, validate=True),
            expected_payload_sha256.encode("ascii"),
        )
    except (ValueError, InvalidSignature):
        return False
    return True


class OwnerRoles(StrictModel):
    data_owner: str = Field(min_length=3, max_length=128)
    security_owner: str = Field(min_length=3, max_length=128)
    storage_owner: str = Field(min_length=3, max_length=128)
    adjudication_owner: str = Field(min_length=3, max_length=128)


class PrivacyRetention(StrictModel):
    data_classification: Literal[
        "synthetic",
        "pseudonymized_operational",
        "aggregated_operational",
    ]
    residency_region: str = Field(min_length=2, max_length=64)
    retention_days: int = Field(gt=0, le=3650)
    deletion_process: str = Field(min_length=10, max_length=500)
    direct_identifiers_prohibited: bool = True
    raw_topology_identifiers_prohibited: bool = True

    @model_validator(mode="after")
    def enforce_privacy_boundary(self) -> Self:
        if (
            not self.direct_identifiers_prohibited
            or not self.raw_topology_identifiers_prohibited
        ):
            raise ValueError("IPTV-P0 requires pseudonymized identifiers")
        return self


class IptvP0Charter(StrictModel):
    schema_version: Literal["0.8.0"] = SCHEMA_VERSION
    charter_id: str = Field(min_length=3, max_length=128)
    managed_live_multicast_only: bool = True
    included_service_path: tuple[str, ...] = Field(min_length=5)
    explicit_exclusions: tuple[str, ...] = Field(min_length=8)
    protected_operator_tasks: tuple[OperatorTask, ...] = Field(min_length=5)
    label_kinds: tuple[LabelKind, ...] = Field(min_length=3)
    claim_ceiling: Literal[
        ClaimCeiling.ENGINEERING_READINESS_ONLY
    ] = ClaimCeiling.ENGINEERING_READINESS_ONLY
    signed_scope_amendment_required: bool = True
    synthetic_cannot_satisfy_real_data_gate: bool = True

    @model_validator(mode="after")
    def freeze_charter(self) -> Self:
        required_path = {
            "source_headend",
            "core_aggregation",
            "multicast_control",
            "transport",
            "access_handoff",
        }
        if not required_path.issubset(self.included_service_path):
            raise ValueError("The complete IPTV-P0 service path is required")
        exclusions = {item.lower() for item in self.explicit_exclusions}
        required_exclusions = {
            "ott_abr",
            "drm",
            "model_training",
            "threshold_tuning",
            "gnn",
            "autonomous_remediation",
            "live_polling",
            "write_credentials",
        }
        if not required_exclusions.issubset(exclusions):
            raise ValueError("The charter is missing a required exclusion")
        if set(self.protected_operator_tasks) != set(OperatorTask):
            raise ValueError("Every protected operator task must be frozen")
        if set(self.label_kinds) != set(LabelKind):
            raise ValueError("Outcome labels must remain separate")
        if (
            not self.signed_scope_amendment_required
            or not self.synthetic_cannot_satisfy_real_data_gate
        ):
            raise ValueError("Charter safety rules cannot be disabled")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class DomainPack(StrictModel):
    schema_version: Literal["0.8.0"] = SCHEMA_VERSION
    pack_id: str = Field(min_length=3, max_length=128)
    pack_version: str = Field(pattern=r"^0\.8\.[0-9]+$")
    issued_at_utc: datetime
    charter: IptvP0Charter
    metric_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    topology_ontology_sha256: str = Field(pattern=SHA256_PATTERN)
    outcome_semantics_sha256: str = Field(pattern=SHA256_PATTERN)
    applicability_profile_sha256: str = Field(pattern=SHA256_PATTERN)
    support_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    signature: SignatureMetadata

    @field_validator("issued_at_utc")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def validate_pack(self) -> Self:
        assert_no_direct_identifiers(self.model_dump(mode="json"))
        return self

    def content_hash(self) -> str:
        value = self.model_dump(mode="json", exclude={"signature"})
        return canonical_hash(value)

    @property
    def signature_valid(self) -> bool:
        return False


class DeploymentPack(StrictModel):
    schema_version: Literal["0.8.0"] = SCHEMA_VERSION
    pack_id: str = Field(min_length=3, max_length=128)
    pack_version: str = Field(pattern=r"^0\.8\.[0-9]+$")
    deployment_pseudonym: str = Field(
        min_length=12,
        max_length=128,
        pattern=r"^dep_[0-9a-f]{16,64}$",
    )
    environment: DeploymentEnvironment
    collection_modes: tuple[CollectionMode, ...] = Field(min_length=1)
    approved_source_ids: tuple[str, ...] = Field(min_length=1)
    domain_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    compatible_frozen_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    ownership: OwnerRoles
    privacy_retention: PrivacyRetention
    real_operator_data: bool
    lawful_scope_approved: bool
    operator_approved: bool
    signature: SignatureMetadata

    @model_validator(mode="after")
    def validate_deployment_pack(self) -> Self:
        if len(set(self.approved_source_ids)) != len(
            self.approved_source_ids
        ):
            raise ValueError("Approved source identifiers must be unique")
        if any(
            mode
            not in {CollectionMode.OFFLINE_JSONL, CollectionMode.OFFLINE_CSV}
            for mode in self.collection_modes
        ):
            raise ValueError("Only offline collection modes are permitted")
        if self.real_operator_data:
            if self.privacy_retention.data_classification == "synthetic":
                raise ValueError("Real data cannot be classified as synthetic")
        else:
            if self.privacy_retention.data_classification != "synthetic":
                raise ValueError("Synthetic packs require synthetic classification")
            if self.signature.state is SignatureState.VERIFIED:
                raise ValueError("Synthetic packs cannot satisfy the real gate")
        assert_no_direct_identifiers(self.model_dump(mode="json"))
        return self

    @property
    def real_gate_identity_complete(self) -> bool:
        return (
            self.real_operator_data
            and self.lawful_scope_approved
            and self.operator_approved
        )

    def content_hash(self) -> str:
        value = self.model_dump(mode="json", exclude={"signature"})
        return canonical_hash(value)
