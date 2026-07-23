from __future__ import annotations

import json
from pathlib import Path

from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
)

from .baseline_evaluator import (
    NoFeasibleThresholdError,
    choose_abstention_threshold,
    evaluate_split,
    load_split_windows,
)
from .models import (
    BaselineEvaluationResult,
    BaselineName,
)
from .result_store import (
    save_evaluation_result,
)


def load_benchmark_fingerprint(
    path: Path,
) -> str:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        record = json.load(file)

    return str(
        record["fingerprint"]["benchmark_sha256"]
    )


def main() -> None:
    nrim_dir = (
        Path(__file__).resolve().parent.parent
    )

    dataset_root = (
        nrim_dir
        / "examples"
        / "simulation"
        / "day09_model_ready"
    )

    manifest_path = (
        dataset_root / "manifest.json"
    )

    freeze_path = (
        dataset_root
        / "benchmark_v0_1_0"
        / "benchmark_freeze.json"
    )

    output_dir = (
        dataset_root
        / "benchmark_v0_1_0"
        / "day11_baselines"
    )

    manifest = load_model_ready_manifest(
        manifest_path
    )

    benchmark_fingerprint = (
        load_benchmark_fingerprint(
            freeze_path
        )
    )

    validation_windows = load_split_windows(
        manifest=manifest,
        split="validation",
    )

    test_windows = load_split_windows(
        manifest=manifest,
        split="test",
    )

    ood_windows = load_split_windows(
        manifest=manifest,
        split="ood_test",
    )

    baselines = list(BaselineName)

    summary_records = []

    for baseline in baselines:
        print(
            f"\nEvaluating {baseline.value}"
        )

        try:
            threshold = choose_abstention_threshold(
                validation_windows=(
                    validation_windows
                ),
                baseline=baseline,
                maximum_healthy_false_selection_rate=0.10,
                minimum_faulty_coverage=0.70,
                random_seed=42,
            )
            threshold_feasible = True
        except NoFeasibleThresholdError:
            threshold = None
            threshold_feasible = False
            print(
                "No abstention threshold satisfies the "
                "validation constraints; reporting the "
                "unabstained baseline."
            )

        _, validation_summary = evaluate_split(
            windows=validation_windows,
            baseline=baseline,
            abstention_threshold=threshold,
            random_seed=42,
        )

        _, test_summary = evaluate_split(
            windows=test_windows,
            baseline=baseline,
            abstention_threshold=threshold,
            random_seed=42,
        )

        _, ood_summary = evaluate_split(
            windows=ood_windows,
            baseline=baseline,
            abstention_threshold=threshold,
            random_seed=42,
        )

        result = BaselineEvaluationResult(
            baseline=baseline,
            benchmark_fingerprint=(
                benchmark_fingerprint
            ),
            abstention_threshold=threshold,
            validation=validation_summary,
            test=test_summary,
            ood_test=ood_summary,
            configuration={
                "random_seed": 42,
                "maximum_healthy_false_selection_rate": 0.10,
                "minimum_faulty_coverage": 0.70,
                "abstention_threshold_feasible": (
                    threshold_feasible
                ),
            },
        )

        json_path, markdown_path = (
            save_evaluation_result(
                result=result,
                output_dir=output_dir,
            )
        )

        summary_records.append(
            result.model_dump(mode="json")
        )

        print(
            "Validation MRR:",
            validation_summary.mrr,
        )
        print(
            "Test MRR:",
            test_summary.mrr,
        )
        print(
            "OOD MRR:",
            ood_summary.mrr,
        )
        print("Saved:", json_path)
        print("Saved:", markdown_path)

    summary_path = output_dir / "summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_records,
            file,
            indent=2,
        )

    print("\nSaved summary:", summary_path)


if __name__ == "__main__":
    main()
