from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BenchmarkStatus(str, Enum):
    DRAFT = "draft"
    REJECTED = "rejected"
    FROZEN = "frozen"
    SUPERSEDED = "superseded"
    PRODUCTION_VALIDATED = "production_validated"


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditFinding(BaseModel):
    code: str = Field(min_length=1)
    severity: AuditSeverity
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class WindowSummary(BaseModel):
    window_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    path: str = Field(min_length=1)

    node_count: int = Field(gt=0)
    edge_count: int = Field(ge=0)
    node_feature_count: int = Field(gt=0)
    edge_feature_count: int = Field(ge=0)

    failure_type: str = Field(min_length=1)
    current_incident: int = Field(ge=0, le=1)
    future_incident: int = Field(ge=0, le=1)

    root_cause_positive_count: int = Field(ge=0)
    affected_service_positive_count: int = Field(ge=0)

    missing_feature_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )


class SplitSummary(BaseModel):
    split: str
    window_count: int = Field(ge=0)

    failure_type_counts: dict[str, int]
    future_incident_counts: dict[str, int]
    current_incident_counts: dict[str, int]

    mean_node_count: float = Field(ge=0.0)
    mean_edge_count: float = Field(ge=0.0)
    mean_missing_feature_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )


class DatasetFingerprint(BaseModel):
    fingerprint_version: str = "1.0.0"

    dataset_manifest_sha256: str = Field(
        min_length=64,
        max_length=64,
    )
    feature_schema_sha256: str = Field(
        min_length=64,
        max_length=64,
    )
    ordered_window_hash_sha256: str = Field(
        min_length=64,
        max_length=64,
    )
    benchmark_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    window_file_hashes: dict[str, str]


class ShortcutResult(BaseModel):
    shortcut_name: str
    target_name: str

    baseline_score: float = Field(ge=0.0, le=1.0)
    shortcut_score: float = Field(ge=0.0, le=1.0)
    score_improvement: float

    threshold: float | None = None
    direction: str | None = None

    suspicious: bool
    explanation: str


class BenchmarkFreezeRecord(BaseModel):
    benchmark_name: str
    benchmark_version: str

    domain: str
    status: BenchmarkStatus

    created_at: datetime = Field(default_factory=utc_now)

    source_manifest_path: str
    dataset_card_path: str
    evaluation_protocol_path: str

    fingerprint: DatasetFingerprint
    split_summaries: list[SplitSummary]

    findings: list[AuditFinding]
    shortcut_results: list[ShortcutResult]

    limitations: list[str]

    @model_validator(mode="after")
    def validate_freeze(
        self,
    ) -> "BenchmarkFreezeRecord":
        critical_findings = [
            finding
            for finding in self.findings
            if finding.severity
            in {
                AuditSeverity.ERROR,
                AuditSeverity.CRITICAL,
            }
        ]

        if (
            self.status == BenchmarkStatus.FROZEN
            and critical_findings
        ):
            raise ValueError(
                "A benchmark with errors cannot be frozen."
            )

        if (
            self.status
            == BenchmarkStatus.PRODUCTION_VALIDATED
            and self.domain == "iptv_synthetic"
        ):
            raise ValueError(
                "Synthetic data cannot be marked "
                "production-validated."
            )

        return self
