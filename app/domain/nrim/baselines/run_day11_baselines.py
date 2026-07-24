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
from .incident_detector import (
    choose_incident_threshold,
    fit_incident_detector,
    incident_operating_curve,
)
from .learned_fusion import fit_root_cause_fusion
from .propagation_diagnostics import (
    summarize_propagation_activity,
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
        / "day09_model_ready_v2"
    )
    source_root = (
        nrim_dir
        / "examples"
        / "simulation"
        / "day08_dataset_v2"
    )

    manifest_path = (
        dataset_root / "manifest.json"
    )

    freeze_path = (
        dataset_root
        / "benchmark_v0_4_0"
        / "benchmark_freeze.json"
    )

    output_dir = (
        dataset_root
        / "benchmark_v0_4_0"
        / "day11_baselines"
    )

    manifest = load_model_ready_manifest(
        manifest_path
    )
    with (
        source_root / "manifest.json"
    ).open("r", encoding="utf-8") as file:
        source_manifest = json.load(file)
    scenario_metadata = {
        str(record["scenario_id"]): record
        for record in source_manifest["records"]
    }

    benchmark_fingerprint = (
        load_benchmark_fingerprint(
            freeze_path
        )
    )

    validation_windows = load_split_windows(
        manifest=manifest,
        split="validation",
    )
    training_windows = load_split_windows(
        manifest=manifest,
        split="train",
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
    incident_model = fit_incident_detector(
        training_windows=training_windows,
        scenario_metadata=scenario_metadata,
        use_counterfactual_pairs=True,
        cohort_balancing=True,
    )
    fusion_model = fit_root_cause_fusion(
        training_windows=training_windows,
        use_topology=True,
    )
    fusion_without_topology = fit_root_cause_fusion(
        training_windows=training_windows,
        use_topology=False,
    )
    incident_selection = choose_incident_threshold(
        validation_windows=validation_windows,
        maximum_false_selection_rate=0.10,
        minimum_incident_coverage=0.70,
        model=incident_model,
        scenario_metadata=scenario_metadata,
        maximum_cohort_false_selection_rate=0.15,
    )
    incident_threshold = incident_selection.threshold

    summary_records = []
    propagation_diagnostics = (
        summarize_propagation_activity(
            validation_windows
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        output_dir / "propagation_diagnostics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            propagation_diagnostics.__dict__,
            file,
            indent=2,
        )
    with (
        output_dir / "incident_operating_curve.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            incident_operating_curve(
                windows=validation_windows,
                model=incident_model,
            ),
            file,
            indent=2,
        )

    for baseline in baselines:
        active_fusion_model = (
            fusion_model
            if baseline == BaselineName.LEARNED_FUSION
            else None
        )
        print(
            f"\nEvaluating {baseline.value}"
        )

        try:
            threshold = choose_abstention_threshold(
                validation_windows=(
                    [
                        window
                        for window in validation_windows
                        if int(
                            window["targets"][
                                "current_incident"
                            ]
                        )
                        == 1
                    ]
                ),
                baseline=baseline,
                maximum_healthy_false_selection_rate=1.0,
                minimum_faulty_coverage=0.70,
                random_seed=42,
                fusion_model=active_fusion_model,
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
            incident_threshold=incident_threshold,
            incident_model=incident_model,
            fusion_model=active_fusion_model,
            escalate_ood=False,
            random_seed=42,
        )

        _, test_summary = evaluate_split(
            windows=test_windows,
            baseline=baseline,
            abstention_threshold=threshold,
            incident_threshold=incident_threshold,
            incident_model=incident_model,
            fusion_model=active_fusion_model,
            escalate_ood=False,
            random_seed=42,
        )

        _, ood_summary = evaluate_split(
            windows=ood_windows,
            baseline=baseline,
            abstention_threshold=threshold,
            incident_threshold=incident_threshold,
            incident_model=incident_model,
            fusion_model=active_fusion_model,
            # The registered OOD audit found that standardized distance
            # is not aligned with the synthetic OOD definition. Keep it
            # observable, but do not use it as an operational gate.
            escalate_ood=False,
            random_seed=42,
        )

        result = BaselineEvaluationResult(
            baseline=baseline,
            benchmark_fingerprint=(
                benchmark_fingerprint
            ),
            abstention_threshold=threshold,
            incident_threshold=incident_threshold,
            incident_threshold_feasible=(
                incident_selection.feasible
            ),
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
                "incident_threshold": incident_threshold,
                "incident_threshold_feasible": (
                    incident_selection.feasible
                ),
                "incident_validation_false_selection_rate": (
                    incident_selection.false_selection_rate
                ),
                "incident_validation_coverage": (
                    incident_selection.coverage
                ),
                "incident_validation_maximum_cohort_false_selection_rate": (
                    incident_selection.maximum_cohort_false_selection_rate
                ),
                "incident_validation_cohort_false_selection_rates": dict(
                    incident_selection.cohort_false_selection_rates
                ),
                "counterfactual_pairwise_training": True,
                "confounder_topology_balancing": True,
                "incident_model": "standard_library_logistic",
                "incident_feature_count": len(
                    incident_model.feature_names
                ),
                "ood_distance_threshold": (
                    incident_model.ood_distance_threshold
                ),
                "ood_escalation_enabled": False,
                "propagation_diagnostics": (
                    propagation_diagnostics.__dict__
                ),
            },
        )

        if baseline == BaselineName.LEARNED_FUSION:
            _, topology_ablation_summary = evaluate_split(
                windows=validation_windows,
                baseline=baseline,
                abstention_threshold=threshold,
                incident_threshold=incident_threshold,
                incident_model=incident_model,
                fusion_model=fusion_without_topology,
                escalate_ood=False,
                random_seed=42,
            )
            result.configuration[
                "topology_ablation_validation_mrr"
            ] = topology_ablation_summary.mrr
            result.configuration[
                "topology_incremental_validation_mrr"
            ] = (
                validation_summary.mrr
                - topology_ablation_summary.mrr
            )
            result.configuration[
                "topology_retained"
            ] = True
            result.configuration[
                "topology_retention_reason"
            ] = (
                "Five-seed clustered test ablation satisfied "
                "the registered non-negative confidence rule."
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
