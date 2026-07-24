from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
)

from .baseline_evaluator import load_split_windows
from .diagnostic_experiments import (
    independent_ranking_evaluation,
    ood_audit,
    stage1_diagnostics,
    stage1_ablation_suite,
    topology_validation,
)
from .learned_fusion import fit_root_cause_fusion


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_report(results: dict[str, Any]) -> str:
    stage1 = results["stage1"]
    selection = stage1["selection"]
    topology = results["topology_validation"]
    ood = results["ood_audit"]
    ranking = results["independent_ranking"]
    learned = ranking["learned_fusion"]

    lines = [
        "# NRIM Focused Diagnostic Experiment",
        "",
        "## Stage 1",
        "",
        (
            f"- Selected threshold: `{selection['threshold']:.6f}`"
        ),
        f"- Feasible: `{selection['feasible']}`",
        (
            "- Stage 1 release status: "
            f"`{'frozen' if selection['feasible'] else 'candidate'}`"
        ),
        (
            "- Validation healthy false-selection: "
            f"`{selection['healthy_false_selection_rate']:.4f}`"
        ),
        (
            "- Maximum confounder-family false-selection: "
            f"`{selection.get('maximum_cohort_false_selection_rate', 0.0):.4f}`"
        ),
        f"- Validation recall: `{selection['recall']:.4f}`",
        f"- Validation precision: `{selection['precision']:.4f}`",
        "",
        "## Ranking without Stage 1 gating",
        "",
        "| Split | All-faulty MRR | Accepted-faulty MRR | Stage 1 recall | End-to-end Hits@1 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("validation", "test", "ood_test"):
        row = learned[split]
        lines.append(
            f"| {split} "
            f"| {row['all_true_faulty']['mrr']:.4f} "
            f"| {row['stage1_accepted_true_faulty']['mrr']:.4f} "
            f"| {row['stage1_recall']:.4f} "
            f"| {row['end_to_end_hits_at_1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Topology validation",
            "",
            f"- Seeds: `{topology['seeds']}`",
            (
                "- Retain topology under the registered rule: "
                f"`{topology['retain_topology']}`"
            ),
        ]
    )
    for split, row in topology[
        "clustered_bootstrap_by_split"
    ].items():
        lines.append(
            f"- {split}: mean delta RR `{row['mean']:.5f}`, "
            f"95% clustered CI "
            f"`[{row['ci_lower']:.5f}, {row['ci_upper']:.5f}]`, "
            f"P(delta>0) `{row['probability_positive']:.3f}`"
        )

    lines.extend(
        [
            "",
            "## OOD audit",
            "",
            f"- OOD AUROC: `{ood['ood_auroc']:.4f}`",
            (
                "- OOD average precision: "
                f"`{ood['ood_average_precision']:.4f}`"
            ),
            (
                "- KS(OOD, train): "
                f"`{ood['ks_ood_vs_train']:.4f}`"
            ),
            (
                "- Distance aligned with OOD definition: "
                f"`{ood['aligned_with_ood_definition']}`"
            ),
            "",
            "Detailed cohorts, feature contributions, drift, "
            "calibration, ablations, and operating curves are "
            "available in `diagnostic_experiments.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    nrim_dir = Path(__file__).resolve().parent.parent
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
    output_dir = (
        dataset_root
        / "benchmark_v0_4_0"
        / "diagnostic_experiments"
    )
    model_manifest = load_model_ready_manifest(
        dataset_root / "manifest.json"
    )
    source_manifest = _load_json(
        source_root / "manifest.json"
    )
    scenario_metadata = {
        str(record["scenario_id"]): record
        for record in source_manifest["records"]
    }
    windows_by_split = {
        split: load_split_windows(
            manifest=model_manifest,
            split=split,
        )
        for split in (
            "train",
            "validation",
            "test",
            "ood_test",
        )
    }

    print("Running Stage 1 diagnostics")
    incident_model, stage1 = stage1_diagnostics(
        windows_by_split=windows_by_split,
        scenario_metadata=scenario_metadata,
    )
    threshold = float(stage1["selection"]["threshold"])
    print("Running Stage 1 ablations")
    stage1["ablations"] = stage1_ablation_suite(
        windows_by_split=windows_by_split,
        scenario_metadata=scenario_metadata,
        full_model=incident_model,
    )

    print("Fitting ranking model")
    fusion_model = fit_root_cause_fusion(
        training_windows=windows_by_split["train"],
        # Operational ranking follows the registered retention gate.
        use_topology=True,
    )
    print("Running independent ranking evaluation")
    independent_ranking = independent_ranking_evaluation(
        windows_by_split=windows_by_split,
        incident_model=incident_model,
        incident_threshold=threshold,
        fusion_model=fusion_model,
    )

    print("Running five-seed topology validation")
    topology = topology_validation(
        training_windows=windows_by_split["train"],
        windows_by_split=windows_by_split,
        scenario_metadata=scenario_metadata,
    )

    print("Auditing OOD distance")
    ood = ood_audit(
        windows_by_split=windows_by_split,
        model=incident_model,
        scenario_metadata=scenario_metadata,
    )

    results = {
        "protocol_version": "1.0.0",
        "cluster_unit": "topology_fingerprint",
        "stage1_release_status": (
            "frozen"
            if stage1["selection"]["feasible"]
            else "candidate"
        ),
        "stage1": stage1,
        "independent_ranking": independent_ranking,
        "topology_validation": topology,
        "ood_audit": ood,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnostic_experiments.json"
    report_path = output_dir / "diagnostic_report.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    report_path.write_text(
        _build_report(results),
        encoding="utf-8",
    )
    print("Saved:", json_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
