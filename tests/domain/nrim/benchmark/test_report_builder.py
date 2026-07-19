from app.domain.nrim.benchmark.models import (
    BenchmarkFreezeRecord,
    BenchmarkStatus,
    DatasetFingerprint,
    SplitSummary,
)
from app.domain.nrim.benchmark.report_builder import (
    build_markdown_report,
)


def test_report_contains_status_and_fingerprint() -> None:
    fingerprint_value = "a" * 64

    record = BenchmarkFreezeRecord(
        benchmark_name="Test Benchmark",
        benchmark_version="0.1.0",
        domain="iptv_synthetic",
        status=BenchmarkStatus.FROZEN,
        source_manifest_path="manifest.json",
        dataset_card_path="card.md",
        evaluation_protocol_path="protocol.md",
        fingerprint=DatasetFingerprint(
            dataset_manifest_sha256=(
                fingerprint_value
            ),
            feature_schema_sha256=(
                fingerprint_value
            ),
            ordered_window_hash_sha256=(
                fingerprint_value
            ),
            benchmark_sha256=(
                fingerprint_value
            ),
            window_file_hashes={},
        ),
        split_summaries=[
            SplitSummary(
                split="train",
                window_count=10,
                failure_type_counts={
                    "healthy": 5,
                    "storage_io_degradation": 5,
                },
                future_incident_counts={
                    "0": 5,
                    "1": 5,
                },
                current_incident_counts={
                    "0": 10
                },
                mean_node_count=12.0,
                mean_edge_count=11.0,
                mean_missing_feature_fraction=0.1,
            )
        ],
        findings=[],
        shortcut_results=[],
        limitations=["Synthetic data."],
    )

    report = build_markdown_report(record)

    assert "FROZEN" in report
    assert fingerprint_value in report
    assert "Synthetic data." in report
