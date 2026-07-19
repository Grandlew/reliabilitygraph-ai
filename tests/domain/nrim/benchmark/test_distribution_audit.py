from app.domain.nrim.benchmark.distribution_audit import (
    audit_root_cause_integrity,
    run_distribution_audit,
)
from app.domain.nrim.benchmark.models import (
    WindowSummary,
)


def make_summary(
    *,
    window_id: str,
    split: str,
    failure_type: str,
    future_incident: int,
    root_positive_count: int,
) -> WindowSummary:
    return WindowSummary(
        window_id=window_id,
        split=split,
        path="window.json",
        node_count=10,
        edge_count=9,
        node_feature_count=20,
        edge_feature_count=8,
        failure_type=failure_type,
        current_incident=0,
        future_incident=future_incident,
        root_cause_positive_count=(
            root_positive_count
        ),
        affected_service_positive_count=0,
        missing_feature_fraction=0.1,
    )


def test_invalid_root_cause_count_is_detected() -> None:
    summaries = [
        make_summary(
            window_id="window_1",
            split="train",
            failure_type="healthy",
            future_incident=0,
            root_positive_count=1,
        )
    ]

    findings = audit_root_cause_integrity(
        summaries
    )

    assert findings


def test_distribution_audit_returns_split_summaries() -> None:
    summaries = []

    failure_types = [
        "healthy",
        "storage_capacity_saturation",
        "storage_io_degradation",
        "cleanup_job_failure",
        "catchup_worker_failure",
    ]

    for split in (
        "train",
        "validation",
        "test",
        "ood_test",
    ):
        for index, failure_type in enumerate(
            failure_types
        ):
            summaries.append(
                make_summary(
                    window_id=(
                        f"{split}_{index}_a"
                    ),
                    split=split,
                    failure_type=failure_type,
                    future_incident=0,
                    root_positive_count=(
                        0
                        if failure_type == "healthy"
                        else 1
                    ),
                )
            )

            summaries.append(
                make_summary(
                    window_id=(
                        f"{split}_{index}_b"
                    ),
                    split=split,
                    failure_type=failure_type,
                    future_incident=1,
                    root_positive_count=(
                        0
                        if failure_type == "healthy"
                        else 1
                    ),
                )
            )

    split_summaries, findings = (
        run_distribution_audit(
            summaries
        )
    )

    assert len(split_summaries) == 4

    blocking = [
        finding
        for finding in findings
        if finding.severity.value
        in {
            "error",
            "critical",
        }
    ]

    assert blocking == []
