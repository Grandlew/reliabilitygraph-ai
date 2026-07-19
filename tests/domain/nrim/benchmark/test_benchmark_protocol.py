from app.domain.nrim.benchmark.benchmark_protocol import (
    determine_benchmark_status,
)
from app.domain.nrim.benchmark.models import (
    AuditFinding,
    AuditSeverity,
    BenchmarkStatus,
    ShortcutResult,
)


def test_error_rejects_benchmark() -> None:
    status = determine_benchmark_status(
        findings=[
            AuditFinding(
                code="ERROR",
                severity=AuditSeverity.ERROR,
                message="Blocking failure.",
            )
        ],
        shortcut_results=[],
    )

    assert status == BenchmarkStatus.REJECTED


def test_clean_benchmark_is_frozen() -> None:
    status = determine_benchmark_status(
        findings=[],
        shortcut_results=[],
    )

    assert status == BenchmarkStatus.FROZEN


def test_suspicious_shortcut_rejects_benchmark() -> None:
    result = ShortcutResult(
        shortcut_name="node_count",
        target_name="future_incident",
        baseline_score=0.5,
        shortcut_score=0.9,
        score_improvement=0.4,
        threshold=50.0,
        direction="greater_equal",
        suspicious=True,
        explanation="Leakage.",
    )

    status = determine_benchmark_status(
        findings=[],
        shortcut_results=[result],
    )

    assert status == BenchmarkStatus.REJECTED
