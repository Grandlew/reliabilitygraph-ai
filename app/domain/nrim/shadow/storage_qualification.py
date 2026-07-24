from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StorageBackendKind(str, Enum):
    SQLITE_TEST_ADAPTER = "sqlite_test_adapter"
    POSTGRESQL = "postgresql"
    RETENTION_LOCKED_OBJECT_STORE = "retention_locked_object_store"
    OTHER_PILOT_INFRASTRUCTURE = "other_pilot_infrastructure"


class StorageQualificationEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    qualification_id: str = Field(min_length=1, max_length=128)
    tested_at_utc: datetime
    backend_kind: StorageBackendKind
    infrastructure_identity: str = Field(min_length=3, max_length=512)
    independently_administered: bool
    concurrent_process_count: int = Field(ge=2)
    concurrent_append_passed: bool
    conflicting_duplicate_test_passed: bool
    transaction_atomicity_test_passed: bool
    retention_lock_enabled: bool
    retention_policy_days: int = Field(gt=0)
    deletion_denial_test_passed: bool
    encrypted_backup_passed: bool
    restore_test_passed: bool
    restored_chain_matches: bool
    immutable_audit_export_passed: bool
    access_control_review_passed: bool
    disaster_recovery_owner: str = Field(min_length=12, max_length=128)
    evidence_artifact_hashes: dict[str, str] = Field(min_length=5)

    @field_validator("tested_at_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Storage test time must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("evidence_artifact_hashes")
    @classmethod
    def hashes_are_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value.values()
        ):
            raise ValueError("Storage evidence hashes must be SHA-256 hex")
        return value


def qualify_storage(
    evidence: StorageQualificationEvidence,
) -> dict[str, Any]:
    criteria = {
        "pilot_backend": (
            evidence.backend_kind
            is not StorageBackendKind.SQLITE_TEST_ADAPTER
        ),
        "independent_administration": evidence.independently_administered,
        "concurrent_append": (
            evidence.concurrent_process_count >= 2
            and evidence.concurrent_append_passed
            and evidence.conflicting_duplicate_test_passed
            and evidence.transaction_atomicity_test_passed
        ),
        "retention": (
            evidence.retention_lock_enabled
            and evidence.deletion_denial_test_passed
        ),
        "backup_restore": (
            evidence.encrypted_backup_passed
            and evidence.restore_test_passed
            and evidence.restored_chain_matches
        ),
        "immutable_audit_export": (
            evidence.immutable_audit_export_passed
        ),
        "access_control": evidence.access_control_review_passed,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "backend_kind": evidence.backend_kind.value,
        "truth_note": (
            "SQLite remains a deterministic test adapter and cannot satisfy "
            "the pilot-storage gate regardless of local unit-test results."
        ),
    }
