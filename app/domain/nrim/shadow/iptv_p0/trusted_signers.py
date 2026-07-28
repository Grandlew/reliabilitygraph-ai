from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..hashing import canonical_hash
from .contracts import SignatureMetadata, utc, verify_signature


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class SignerRole(str, Enum):
    DOMAIN_AUTHORITY = "domain_authority"
    DEPLOYMENT_AUTHORITY = "deployment_authority"
    REPLAY_GOVERNANCE = "replay_governance"
    QUALIFICATION_AUTHORITY = "qualification_authority"


class TrustedSigner(StrictModel):
    signer_id: str = Field(pattern=r"^signer_[0-9a-z_.-]{4,120}$")
    key_id: str = Field(min_length=3, max_length=128)
    public_key_base64: str = Field(min_length=40, max_length=128)
    roles: tuple[SignerRole, ...] = Field(min_length=1)
    deployment_pseudonym: str | None = Field(
        default=None,
        pattern=r"^dep_[0-9a-f]{16,64}$",
    )
    valid_from_utc: datetime
    valid_to_utc: datetime
    revoked_at_utc: datetime | None = None

    @field_validator("valid_from_utc", "valid_to_utc", "revoked_at_utc")
    @classmethod
    def normalize_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else utc(value)

    @field_validator("public_key_base64")
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError("Trusted signer public key is not base64") from error
        if len(decoded) != 32:
            raise ValueError("Trusted signer Ed25519 key must be 32 bytes")
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if self.valid_to_utc <= self.valid_from_utc:
            raise ValueError("Trusted signer validity interval must be positive")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("Trusted signer roles must be unique")
        if (
            any(
                role is not SignerRole.DOMAIN_AUTHORITY
                for role in self.roles
            )
            and self.deployment_pseudonym is None
        ):
            raise ValueError(
                "Deployment, replay and qualification signers require "
                "deployment binding"
            )
        if (
            self.revoked_at_utc is not None
            and self.revoked_at_utc < self.valid_from_utc
        ):
            raise ValueError("Signer revocation predates its validity")
        return self

    def authorized_at(
        self,
        *,
        role: SignerRole,
        deployment_pseudonym: str,
        at_utc: datetime,
    ) -> bool:
        moment = utc(at_utc)
        if role not in self.roles:
            return False
        if not (self.valid_from_utc <= moment < self.valid_to_utc):
            return False
        if self.revoked_at_utc is not None and moment >= self.revoked_at_utc:
            return False
        if (
            self.deployment_pseudonym is not None
            and self.deployment_pseudonym != deployment_pseudonym
        ):
            return False
        return True


class TrustedSignerRegistry(StrictModel):
    registry_id: str = Field(min_length=3, max_length=128)
    registry_version: str = Field(pattern=r"^0\.8\.[0-9]+$")
    signers: tuple[TrustedSigner, ...] = ()

    @model_validator(mode="after")
    def unique_signers(self) -> Self:
        signer_ids = [item.signer_id for item in self.signers]
        key_ids = [item.key_id for item in self.signers]
        public_keys = [item.public_key_base64 for item in self.signers]
        if len(signer_ids) != len(set(signer_ids)):
            raise ValueError("Trusted signer identifiers must be unique")
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("Trusted signer key identifiers must be unique")
        if len(public_keys) != len(set(public_keys)):
            raise ValueError(
                "Trusted signer public keys must represent distinct identities"
            )
        return self

    def signer(self, key_id: str | None) -> TrustedSigner | None:
        return next(
            (item for item in self.signers if item.key_id == key_id),
            None,
        )

    def verify(
        self,
        signature: SignatureMetadata,
        *,
        expected_payload_sha256: str,
        required_role: SignerRole,
        deployment_pseudonym: str,
        at_utc: datetime,
    ) -> bool:
        signer = self.signer(signature.key_id)
        if signer is None or not signer.authorized_at(
            role=required_role,
            deployment_pseudonym=deployment_pseudonym,
            at_utc=at_utc,
        ):
            return False
        return verify_signature(
            signature,
            expected_payload_sha256=expected_payload_sha256,
            trusted_public_key_base64=signer.public_key_base64,
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))
