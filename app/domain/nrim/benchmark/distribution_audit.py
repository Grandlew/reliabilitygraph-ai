from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Iterable

from .models import (
    AuditFinding,
    AuditSeverity,
    SplitSummary,
    WindowSummary,
)


EXPECTED_SPLITS = {
    "train",
    "validation",
    "test",
    "ood_test",
}


EXPECTED_FAILURE_TYPES = {
    "healthy",
    "storage_capacity_saturation",
    "storage_io_degradation",
    "cleanup_job_failure",
    "catchup_worker_failure",
}


def summarize_splits(
    summaries: list[WindowSummary],
) -> list[SplitSummary]:
    grouped: dict[
        str,
        list[WindowSummary],
    ] = defaultdict(list)

    for summary in summaries:
        grouped[summary.split].append(summary)

    results: list[SplitSummary] = []

    for split in sorted(grouped):
        windows = grouped[split]

        results.append(
            SplitSummary(
                split=split,
                window_count=len(windows),
                failure_type_counts=dict(
                    Counter(
                        window.failure_type
                        for window in windows
                    )
                ),
                future_incident_counts=dict(
                    Counter(
                        str(window.future_incident)
                        for window in windows
                    )
                ),
                current_incident_counts=dict(
                    Counter(
                        str(window.current_incident)
                        for window in windows
                    )
                ),
                mean_node_count=round(
                    mean(
                        window.node_count
                        for window in windows
                    ),
                    6,
                ),
                mean_edge_count=round(
                    mean(
                        window.edge_count
                        for window in windows
                    ),
                    6,
                ),
                mean_missing_feature_fraction=round(
                    mean(
                        window.missing_feature_fraction
                        for window in windows
                    ),
                    6,
                ),
            )
        )

    return results


def audit_split_presence(
    summaries: list[WindowSummary],
) -> list[AuditFinding]:
    present = {
        summary.split
        for summary in summaries
    }

    missing = sorted(
        EXPECTED_SPLITS - present
    )

    if not missing:
        return []

    return [
        AuditFinding(
            code="MISSING_SPLIT",
            severity=AuditSeverity.ERROR,
            message=(
                "Required dataset splits are missing."
            ),
            details={"missing_splits": missing},
        )
    ]


def audit_failure_classes(
    summaries: list[WindowSummary],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    by_split: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for summary in summaries:
        by_split[summary.split][
            summary.failure_type
        ] += 1

    training_classes = set(
        by_split["train"].keys()
    )

    missing_training = sorted(
        EXPECTED_FAILURE_TYPES
        - training_classes
    )

    if missing_training:
        findings.append(
            AuditFinding(
                code="TRAIN_FAILURE_CLASS_MISSING",
                severity=AuditSeverity.ERROR,
                message=(
                    "Training split does not contain every "
                    "required failure class."
                ),
                details={
                    "missing_classes": missing_training
                },
            )
        )

    for split, counts in by_split.items():
        fault_counts = {
            failure_type: count
            for failure_type, count in counts.items()
            if failure_type != "healthy"
        }

        if not fault_counts:
            continue

        maximum = max(fault_counts.values())
        minimum = min(fault_counts.values())

        if minimum > 0 and maximum / minimum > 4.0:
            findings.append(
                AuditFinding(
                    code="SEVERE_CLASS_IMBALANCE",
                    severity=AuditSeverity.WARNING,
                    message=(
                        f"Failure-class imbalance exceeds "
                        f"4:1 in {split}."
                    ),
                    details={
                        "split": split,
                        "counts": fault_counts,
                    },
                )
            )

    return findings


def audit_binary_targets(
    summaries: list[WindowSummary],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    for split in sorted(
        {
            summary.split
            for summary in summaries
        }
    ):
        split_windows = [
            summary
            for summary in summaries
            if summary.split == split
        ]

        for target_name in (
            "future_incident",
            "current_incident",
        ):
            values = {
                int(getattr(window, target_name))
                for window in split_windows
            }

            if len(values) < 2:
                severity = (
                    AuditSeverity.ERROR
                    if split
                    in {
                        "train",
                        "validation",
                        "test",
                    }
                    else AuditSeverity.WARNING
                )

                findings.append(
                    AuditFinding(
                        code="BINARY_TARGET_SINGLE_CLASS",
                        severity=severity,
                        message=(
                            f"{target_name} contains one class "
                            f"in {split}."
                        ),
                        details={
                            "split": split,
                            "target": target_name,
                            "values": sorted(values),
                        },
                    )
                )

    return findings


def audit_root_cause_integrity(
    summaries: list[WindowSummary],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    invalid = []

    for summary in summaries:
        expected = (
            0
            if summary.failure_type == "healthy"
            else 1
        )

        if (
            summary.root_cause_positive_count
            != expected
        ):
            invalid.append(
                {
                    "window_id": summary.window_id,
                    "failure_type": (
                        summary.failure_type
                    ),
                    "expected": expected,
                    "actual": (
                        summary
                        .root_cause_positive_count
                    ),
                }
            )

    if invalid:
        findings.append(
            AuditFinding(
                code="INVALID_ROOT_CAUSE_TARGET",
                severity=AuditSeverity.CRITICAL,
                message=(
                    "Root-cause targets violate "
                    "single-fault semantics."
                ),
                details={
                    "invalid_windows": invalid[:50],
                    "invalid_count": len(invalid),
                },
            )
        )

    return findings


def audit_missingness_shift(
    split_summaries: Iterable[SplitSummary],
    *,
    warning_difference: float = 0.20,
) -> list[AuditFinding]:
    summaries = {
        item.split: item
        for item in split_summaries
    }

    training = summaries.get("train")

    if training is None:
        return []

    findings: list[AuditFinding] = []

    for split in (
        "validation",
        "test",
    ):
        candidate = summaries.get(split)

        if candidate is None:
            continue

        difference = abs(
            candidate.mean_missing_feature_fraction
            - training.mean_missing_feature_fraction
        )

        if difference > warning_difference:
            findings.append(
                AuditFinding(
                    code="MISSINGNESS_DISTRIBUTION_SHIFT",
                    severity=AuditSeverity.WARNING,
                    message=(
                        f"Missingness differs strongly between "
                        f"train and {split}."
                    ),
                    details={
                        "train_missing_fraction": (
                            training
                            .mean_missing_feature_fraction
                        ),
                        "comparison_split": split,
                        "comparison_missing_fraction": (
                            candidate
                            .mean_missing_feature_fraction
                        ),
                        "absolute_difference": difference,
                    },
                )
            )

    return findings


def run_distribution_audit(
    summaries: list[WindowSummary],
) -> tuple[
    list[SplitSummary],
    list[AuditFinding],
]:
    split_summaries = summarize_splits(
        summaries
    )

    findings = []

    findings.extend(
        audit_split_presence(summaries)
    )
    findings.extend(
        audit_failure_classes(summaries)
    )
    findings.extend(
        audit_binary_targets(summaries)
    )
    findings.extend(
        audit_root_cause_integrity(summaries)
    )
    findings.extend(
        audit_missingness_shift(split_summaries)
    )

    return split_summaries, findings
