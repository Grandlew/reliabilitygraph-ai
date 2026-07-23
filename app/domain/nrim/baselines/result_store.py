from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BaselineEvaluationResult


def save_evaluation_result(
    *,
    result: BaselineEvaluationResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / f"{result.baseline.value}.json"
    )

    markdown_path = (
        output_dir
        / f"{result.baseline.value}.md"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.model_dump(mode="json"),
            file,
            indent=2,
        )

    markdown_path.write_text(
        build_result_markdown(result),
        encoding="utf-8",
    )

    return json_path, markdown_path


def metric_row(
    *,
    split: str,
    summary,
) -> str:
    return (
        f"| {split} "
        f"| {summary.mrr:.4f} "
        f"| {summary.hits_at_1:.4f} "
        f"| {summary.hits_at_3:.4f} "
        f"| {summary.mean_rank or 0.0:.3f} "
        f"| {summary.healthy_false_selection_rate:.4f} "
        f"| {summary.faulty_coverage:.4f} "
        f"| {summary.mean_runtime_ms:.4f} |"
    )


def build_result_markdown(
    result: BaselineEvaluationResult,
) -> str:
    config_json = json.dumps(result.configuration, indent=2, sort_keys=True)
    threshold_display = (
        str(result.abstention_threshold)
        if result.abstention_threshold is not None
        else "disabled (no feasible validation threshold)"
    )

    lines = [
        f"# Evaluation Result: {result.baseline.value}",
        "",
        f"**Benchmark Fingerprint:** `{result.benchmark_fingerprint}`",
        f"**Abstention Threshold:** `{threshold_display}`",
        "",
        "## Metrics",
        "",
        "| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Mean Runtime (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        metric_row(split="Validation", summary=result.validation),
        metric_row(split="Test", summary=result.test),
        metric_row(split="OOD Test", summary=result.ood_test),
        "",
        "## Configuration",
        "```json",
        config_json,
        "```",
        "",
        "> **Note on Synthetic Data:** These metrics are derived from simulated IPTV environments. ",
        "> Performance on real-world production systems may vary based on topology complexity and noise profile."
    ]

    return "\n".join(lines)
