from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BenchmarkFreezeRecord


def build_markdown_report(
    record: BenchmarkFreezeRecord,
) -> str:
    lines = [
        "# NRIM Benchmark Freeze Report",
        "",
        f"**Benchmark:** {record.benchmark_name}",
        "",
        f"**Version:** {record.benchmark_version}",
        "",
        f"**Domain:** {record.domain}",
        "",
        f"**Status:** {record.status.value}",
        "",
        "## Fingerprint",
        "",
        "```text",
        record.fingerprint.benchmark_sha256,
        "```",
        "",
        "## Split Summary",
        "",
        (
            "| Split | Windows | Mean nodes | "
            "Mean edges | Mean missingness |"
        ),
        "|---|---:|---:|---:|---:|",
    ]

    for summary in record.split_summaries:
        lines.append(
            f"| {summary.split} "
            f"| {summary.window_count} "
            f"| {summary.mean_node_count:.2f} "
            f"| {summary.mean_edge_count:.2f} "
            f"| {summary.mean_missing_feature_fraction:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Audit Findings",
            "",
        ]
    )

    if record.findings:
        for finding in record.findings:
            lines.append(
                f"- **{finding.severity.value.upper()} "
                f"{finding.code}:** {finding.message}"
            )
    else:
        lines.append(
            "- No structural or distribution findings."
        )

    lines.extend(
        [
            "",
            "## Shortcut Audit",
            "",
            (
                "| Shortcut | Target | Baseline | "
                "Shortcut | Improvement | Suspicious |"
            ),
            "|---|---|---:|---:|---:|---|",
        ]
    )

    for result in record.shortcut_results:
        lines.append(
            f"| {result.shortcut_name} "
            f"| {result.target_name} "
            f"| {result.baseline_score:.4f} "
            f"| {result.shortcut_score:.4f} "
            f"| {result.score_improvement:.4f} "
            f"| {'yes' if result.suspicious else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in record.limitations:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A frozen status means the synthetic benchmark "
                "passed the defined development audits."
            ),
            "",
            (
                "It does not establish production performance "
                "or synthetic-to-real transfer."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_freeze_record(
    *,
    record: BenchmarkFreezeRecord,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record.model_dump(mode="json"),
            file,
            indent=2,
        )

    markdown_path.write_text(
        build_markdown_report(record),
        encoding="utf-8",
    )
