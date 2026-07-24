from __future__ import annotations

import json
from pathlib import Path

from .benchmark_protocol import (
    BenchmarkAcceptancePolicy,
    determine_benchmark_status,
    shortcut_findings,
)
from .dataset_fingerprint import (
    build_dataset_fingerprint,
)
from .dataset_loader import (
    iter_window_records,
    load_model_ready_manifest,
    load_window_summaries,
)
from .distribution_audit import (
    run_distribution_audit,
)
from .models import (
    AuditFinding,
    AuditSeverity,
    BenchmarkFreezeRecord,
)
from .report_builder import (
    save_freeze_record,
)
from .shortcut_audit import (
    run_root_cause_shortcut_audit,
    run_shortcut_audit,
)


def main() -> None:
    nrim_dir = Path(__file__).resolve().parent.parent

    model_ready_root = (
        nrim_dir
        / "examples"
        / "simulation"
        / "day09_model_ready_v2"
    )

    manifest_path = (
        model_ready_root
        / "manifest.json"
    )

    output_root = (
        model_ready_root
        / "benchmark_v0_4_0"
    )

    manifest = load_model_ready_manifest(
        manifest_path
    )

    summaries = load_window_summaries(
        manifest
    )

    fingerprint = build_dataset_fingerprint(
        manifest=manifest
    )

    split_summaries, findings = (
        run_distribution_audit(
            summaries
        )
    )

    shortcut_results = run_shortcut_audit(
        summaries,
        target_name="future_incident",
    )

    loaded_windows = [
        (str(record["split"]), window)
        for record, window in iter_window_records(manifest)
    ]
    shortcut_results.extend(
        run_root_cause_shortcut_audit(
            training_windows=[
                window
                for split, window in loaded_windows
                if split == "train"
            ],
            validation_windows=[
                window
                for split, window in loaded_windows
                if split == "validation"
            ],
        )
    )

    findings.extend(
        shortcut_findings(
            shortcut_results
        )
    )

    if not summaries:
        findings.append(
            AuditFinding(
                code="EMPTY_BENCHMARK",
                severity=AuditSeverity.CRITICAL,
                message=(
                    "The model-ready benchmark "
                    "contains no windows."
                ),
            )
        )

    policy = BenchmarkAcceptancePolicy(
        allow_warnings=True,
        reject_suspicious_shortcuts=True,
    )

    status = determine_benchmark_status(
        findings=findings,
        shortcut_results=shortcut_results,
        policy=policy,
    )

    freeze_record = BenchmarkFreezeRecord(
        benchmark_name=(
            "NRIM IPTV Reliability Benchmark"
        ),
        benchmark_version="0.4.0",
        domain="iptv_synthetic",
        status=status,
        source_manifest_path=str(
            manifest_path
        ),
        dataset_card_path=(
            "docs/day10_dataset_card.md"
        ),
        evaluation_protocol_path=(
            "docs/day10_evaluation_protocol.md"
        ),
        fingerprint=fingerprint,
        split_summaries=split_summaries,
        findings=findings,
        shortcut_results=shortcut_results,
        limitations=[
            "The benchmark is entirely synthetic.",
            (
                "The failure catalogue is limited to "
                "four initial IPTV fault families."
            ),
            (
                "The operating models are simplified "
                "representations of real deployments."
            ),
            (
                "No real NetUP engineer-confirmed "
                "incident outcomes are included."
            ),
            (
                "The benchmark contains no ISP "
                "broadband scenarios."
            ),
            (
                "Synthetic benchmark performance "
                "does not prove production performance."
            ),
        ],
    )

    save_freeze_record(
        record=freeze_record,
        json_path=(
            output_root
            / "benchmark_freeze.json"
        ),
        markdown_path=(
            output_root
            / "benchmark_report.md"
        ),
    )

    print("\nBENCHMARK STATUS")
    print(status.value)

    print("\nBENCHMARK FINGERPRINT")
    print(
        fingerprint.benchmark_sha256
    )

    print("\nSPLIT SUMMARIES")
    print(
        json.dumps(
            [
                summary.model_dump(
                    mode="json"
                )
                for summary
                in split_summaries
            ],
            indent=2,
        )
    )

    print("\nFINDINGS")
    print(
        json.dumps(
            [
                finding.model_dump(
                    mode="json"
                )
                for finding in findings
            ],
            indent=2,
        )
    )

    print(
        "\nSaved freeze record:",
        output_root / "benchmark_freeze.json",
    )
    print(
        "Saved report:",
        output_root / "benchmark_report.md",
    )

    if status.value == "rejected":
        raise RuntimeError(
            "Benchmark rejected. Correct audit "
            "failures before model training."
        )


if __name__ == "__main__":
    main()
