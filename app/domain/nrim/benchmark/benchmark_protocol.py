from __future__ import annotations

from dataclasses import dataclass

from .models import (
    AuditFinding,
    AuditSeverity,
    BenchmarkStatus,
    ShortcutResult,
)


@dataclass(frozen=True)
class BenchmarkAcceptancePolicy:
    allow_warnings: bool = True
    reject_suspicious_shortcuts: bool = True
    require_all_core_splits: bool = True


def shortcut_findings(
    results: list[ShortcutResult],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    for result in results:
        if not result.suspicious:
            continue

        findings.append(
            AuditFinding(
                code="SUSPICIOUS_SHORTCUT",
                severity=AuditSeverity.ERROR,
                message=(
                    f"Shortcut '{result.shortcut_name}' "
                    f"predicts {result.target_name} too well."
                ),
                details=result.model_dump(
                    mode="json"
                ),
            )
        )

    return findings


def determine_benchmark_status(
    *,
    findings: list[AuditFinding],
    shortcut_results: list[ShortcutResult],
    policy: BenchmarkAcceptancePolicy | None = None,
) -> BenchmarkStatus:
    policy = (
        policy
        or BenchmarkAcceptancePolicy()
    )

    combined_findings = list(findings)

    if policy.reject_suspicious_shortcuts:
        combined_findings.extend(
            shortcut_findings(
                shortcut_results
            )
        )

    blocking = [
        finding
        for finding in combined_findings
        if finding.severity
        in {
            AuditSeverity.ERROR,
            AuditSeverity.CRITICAL,
        }
    ]

    if blocking:
        return BenchmarkStatus.REJECTED

    if (
        not policy.allow_warnings
        and any(
            finding.severity
            == AuditSeverity.WARNING
            for finding in combined_findings
        )
    ):
        return BenchmarkStatus.REJECTED

    return BenchmarkStatus.FROZEN
