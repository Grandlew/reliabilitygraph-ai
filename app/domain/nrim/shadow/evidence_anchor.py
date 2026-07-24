from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Self

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
from .store import AppendOnlyEvidenceStore


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Checkpoint timestamps must include a timezone")
    return value.astimezone(timezone.utc)


class EvidenceCheckpoint(StrictModel):
    checkpoint_version: str = "0.7.1"
    checkpoint_sequence: int = Field(ge=1)
    created_at_utc: datetime
    application_instance_id: str = Field(min_length=3, max_length=128)
    application_trust_domain: str = Field(min_length=3, max_length=256)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_count: int = Field(ge=0)
    previous_evidence_count: int = Field(ge=0)
    prediction_chain_head: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latest_decision_cutoff_utc: datetime | None = None

    @field_validator("created_at_utc", "latest_decision_cutoff_utc")
    @classmethod
    def validate_time(cls, value):
        return _utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.evidence_count < self.previous_evidence_count:
            raise ValueError("Evidence count cannot move backwards")
        if (
            self.checkpoint_sequence == 1
            and self.previous_checkpoint_sha256 != "0" * 64
        ):
            raise ValueError("First checkpoint must use the zero predecessor")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class SignedCheckpoint(StrictModel):
    checkpoint: EvidenceCheckpoint
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_public_key_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    signature_algorithm: str = "ed25519"
    signature_base64: str


class AnchorReceiptPayload(StrictModel):
    receipt_version: str = "0.7.1"
    sink_id: str = Field(min_length=3, max_length=128)
    external_trust_domain: str = Field(min_length=3, max_length=256)
    external_location: str = Field(min_length=3, max_length=1000)
    received_at_utc: datetime
    checkpoint_sequence: int = Field(ge=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signed_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("received_at_utc")
    @classmethod
    def validate_received_time(cls, value: datetime) -> datetime:
        return _utc(value)


class AnchorReceipt(StrictModel):
    payload: AnchorReceiptPayload
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    custodian_public_key_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    signature_algorithm: str = "ed25519"
    signature_base64: str
    adapter_truth_status: str

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ExternalAnchorSink(Protocol):
    def accept(
        self,
        *,
        signed_checkpoint: SignedCheckpoint,
        application_public_key_pem: bytes,
    ) -> AnchorReceipt: ...


def _public_key(public_key_pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Trust anchor must be an Ed25519 public key")
    return key


def verify_signed_checkpoint(
    *,
    signed_checkpoint: SignedCheckpoint,
    application_public_key_pem: bytes,
) -> None:
    checkpoint_payload = signed_checkpoint.checkpoint.model_dump(mode="json")
    if canonical_hash(checkpoint_payload) != (
        signed_checkpoint.checkpoint_sha256
    ):
        raise ValueError("Checkpoint commitment differs")
    if bytes_hash(application_public_key_pem) != (
        signed_checkpoint.application_public_key_sha256
    ):
        raise ValueError("Application trust anchor differs")
    try:
        _public_key(application_public_key_pem).verify(
            base64.b64decode(
                signed_checkpoint.signature_base64,
                validate=True,
            ),
            canonical_json(checkpoint_payload).encode("utf-8"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("Checkpoint signature is invalid") from error


class IndependentFileAnchorSink:
    """Test adapter for a separately keyed append-only checkpoint sink.

    A real deployment must replace the directory with independently
    administered retention-locked storage and keep the custodian key outside
    the application environment.
    """

    def __init__(
        self,
        *,
        directory: Path,
        sink_id: str,
        external_trust_domain: str,
        application_trust_domain: str,
        custodian_private_key: Ed25519PrivateKey,
        custodian_public_key_pem: bytes,
    ) -> None:
        if external_trust_domain == application_trust_domain:
            raise ValueError("Evidence sink must name a different trust domain")
        self.directory = directory
        self.sink_id = sink_id
        self.external_trust_domain = external_trust_domain
        self.application_trust_domain = application_trust_domain
        self.private_key = custodian_private_key
        self.public_key_pem = custodian_public_key_pem
        if (
            custodian_private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            != custodian_public_key_pem
        ):
            raise ValueError("Custodian private/public keys differ")
        self.directory.mkdir(parents=True, exist_ok=True)

    def _existing_records(
        self,
    ) -> list[tuple[SignedCheckpoint, AnchorReceipt]]:
        records = []
        for path in sorted(self.directory.glob("checkpoint_*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                (
                    SignedCheckpoint.model_validate(
                        value["signed_checkpoint"]
                    ),
                    AnchorReceipt.model_validate(value["receipt"]),
                )
            )
        return records

    def accept(
        self,
        *,
        signed_checkpoint: SignedCheckpoint,
        application_public_key_pem: bytes,
    ) -> AnchorReceipt:
        verify_signed_checkpoint(
            signed_checkpoint=signed_checkpoint,
            application_public_key_pem=application_public_key_pem,
        )
        checkpoint = signed_checkpoint.checkpoint
        if checkpoint.application_trust_domain != (
            self.application_trust_domain
        ):
            raise ValueError("Checkpoint application trust domain differs")
        existing = self._existing_records()
        expected_sequence = len(existing) + 1
        if checkpoint.checkpoint_sequence != expected_sequence:
            raise ValueError("External checkpoint sequence is not consecutive")
        expected_checkpoint_predecessor = (
            existing[-1][0].checkpoint_sha256 if existing else "0" * 64
        )
        if checkpoint.previous_checkpoint_sha256 != (
            expected_checkpoint_predecessor
        ):
            raise ValueError("External checkpoint predecessor differs")
        previous_receipt = (
            existing[-1][1].content_hash() if existing else "0" * 64
        )
        payload = AnchorReceiptPayload(
            sink_id=self.sink_id,
            external_trust_domain=self.external_trust_domain,
            external_location=(
                f"{self.external_trust_domain}/"
                f"checkpoint_{checkpoint.checkpoint_sequence:08d}"
            ),
            received_at_utc=datetime.now(timezone.utc),
            checkpoint_sequence=checkpoint.checkpoint_sequence,
            checkpoint_sha256=signed_checkpoint.checkpoint_sha256,
            signed_checkpoint_sha256=canonical_hash(
                signed_checkpoint.model_dump(mode="json")
            ),
            previous_receipt_sha256=previous_receipt,
        )
        payload_json = payload.model_dump(mode="json")
        receipt = AnchorReceipt(
            payload=payload,
            payload_sha256=canonical_hash(payload_json),
            custodian_public_key_sha256=bytes_hash(self.public_key_pem),
            signature_base64=base64.b64encode(
                self.private_key.sign(
                    canonical_json(payload_json).encode("utf-8")
                )
            ).decode("ascii"),
            adapter_truth_status=(
                "filesystem_test_adapter_not_independent_infrastructure"
            ),
        )
        record = {
            "signed_checkpoint": signed_checkpoint.model_dump(mode="json"),
            "receipt": receipt.model_dump(mode="json"),
        }
        path = self.directory / (
            f"checkpoint_{checkpoint.checkpoint_sequence:08d}.json"
        )
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        return receipt


class EvidenceAnchorService:
    def __init__(
        self,
        *,
        store: AppendOnlyEvidenceStore,
        application_instance_id: str,
        application_trust_domain: str,
        protocol_sha256: str,
        model_bundle_sha256: str,
        application_private_key: Ed25519PrivateKey,
        application_public_key_pem: bytes,
    ) -> None:
        self.store = store
        self.application_instance_id = application_instance_id
        self.application_trust_domain = application_trust_domain
        self.protocol_sha256 = protocol_sha256
        self.model_bundle_sha256 = model_bundle_sha256
        self.private_key = application_private_key
        self.public_key_pem = application_public_key_pem
        if (
            application_private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            != application_public_key_pem
        ):
            raise ValueError("Application private/public keys differ")

    def checkpoint(
        self,
        *,
        sink: ExternalAnchorSink,
        sequence: int,
        previous_checkpoint_sha256: str,
        previous_evidence_count: int,
    ) -> tuple[SignedCheckpoint, AnchorReceipt]:
        state = self.store.prediction_chain_state()
        if not state["chain_valid"]:
            raise ValueError("Prediction chain is invalid")
        latest = state["latest_cutoff_utc"]
        checkpoint = EvidenceCheckpoint(
            checkpoint_sequence=sequence,
            created_at_utc=datetime.now(timezone.utc),
            application_instance_id=self.application_instance_id,
            application_trust_domain=self.application_trust_domain,
            protocol_sha256=self.protocol_sha256,
            model_bundle_sha256=self.model_bundle_sha256,
            evidence_count=state["evidence_count"],
            previous_evidence_count=previous_evidence_count,
            prediction_chain_head=state["chain_head"],
            previous_checkpoint_sha256=previous_checkpoint_sha256,
            latest_decision_cutoff_utc=(
                datetime.fromisoformat(latest) if latest else None
            ),
        )
        payload = checkpoint.model_dump(mode="json")
        signed = SignedCheckpoint(
            checkpoint=checkpoint,
            checkpoint_sha256=canonical_hash(payload),
            application_public_key_sha256=bytes_hash(
                self.public_key_pem
            ),
            signature_base64=base64.b64encode(
                self.private_key.sign(
                    canonical_json(payload).encode("utf-8")
                )
            ).decode("ascii"),
        )
        receipt = sink.accept(
            signed_checkpoint=signed,
            application_public_key_pem=self.public_key_pem,
        )
        return signed, receipt


def verify_anchor_receipt(
    *,
    signed_checkpoint: SignedCheckpoint,
    receipt: AnchorReceipt,
    application_public_key_pem: bytes,
    custodian_public_key_pem: bytes,
) -> None:
    verify_signed_checkpoint(
        signed_checkpoint=signed_checkpoint,
        application_public_key_pem=application_public_key_pem,
    )
    payload = receipt.payload.model_dump(mode="json")
    if canonical_hash(payload) != receipt.payload_sha256:
        raise ValueError("External receipt commitment differs")
    if receipt.payload.checkpoint_sha256 != (
        signed_checkpoint.checkpoint_sha256
    ):
        raise ValueError("Receipt names a different checkpoint")
    if bytes_hash(custodian_public_key_pem) != (
        receipt.custodian_public_key_sha256
    ):
        raise ValueError("External custodian trust anchor differs")
    try:
        _public_key(custodian_public_key_pem).verify(
            base64.b64decode(receipt.signature_base64, validate=True),
            canonical_json(payload).encode("utf-8"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("External receipt signature is invalid") from error


def verify_store_matches_checkpoint(
    *,
    store: AppendOnlyEvidenceStore,
    signed_checkpoint: SignedCheckpoint,
) -> bool:
    state = store.prediction_chain_state()
    checkpoint = signed_checkpoint.checkpoint
    return (
        bool(state["chain_valid"])
        and int(state["evidence_count"]) == checkpoint.evidence_count
        and str(state["chain_head"]) == checkpoint.prediction_chain_head
    )
