import pytest
from pydantic import ValidationError

from app.domain.nrim.benchmark.models import (
    AuditFinding,
    AuditSeverity,
    BenchmarkFreezeRecord,
    BenchmarkStatus,
    DatasetFingerprint,
)


def make_fingerprint() -> DatasetFingerprint:
    value = "a" * 64

    return DatasetFingerprint(
        dataset_manifest_sha256=value,
        feature_schema_sha256=value,
        ordered_window_hash_sha256=value,
        benchmark_sha256=value,
        window_file_hashes={},
    )


def test_frozen_benchmark_rejects_errors() -> None:
    with pytest.raises(ValidationError):
        BenchmarkFreezeRecord(
            benchmark_name="test",
            benchmark_version="0.1.0",
            domain="iptv_synthetic",
            status=BenchmarkStatus.FROZEN,
            source_manifest_path="manifest.json",
            dataset_card_path="card.md",
            evaluation_protocol_path="protocol.md",
            fingerprint=make_fingerprint(),
            split_summaries=[],
            findings=[
                AuditFinding(
                    code="ERROR",
                    severity=AuditSeverity.ERROR,
                    message="Blocking error.",
                )
            ],
            shortcut_results=[],
            limitations=["Synthetic."],
        )


def test_synthetic_benchmark_cannot_be_production_validated() -> None:
    with pytest.raises(ValidationError):
        BenchmarkFreezeRecord(
            benchmark_name="test",
            benchmark_version="0.1.0",
            domain="iptv_synthetic",
            status=(
                BenchmarkStatus.PRODUCTION_VALIDATED
            ),
            source_manifest_path="manifest.json",
            dataset_card_path="card.md",
            evaluation_protocol_path="protocol.md",
            fingerprint=make_fingerprint(),
            split_summaries=[],
            findings=[],
            shortcut_results=[],
            limitations=["Synthetic."],
        )
