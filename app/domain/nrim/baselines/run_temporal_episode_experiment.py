from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
)

from .baseline_evaluator import load_split_windows
from .incident_detector import (
    IncidentDetectorModel,
    choose_incident_threshold,
    fit_incident_detector,
    incident_features,
)
from .learned_fusion import RootCauseFusionModel, fit_root_cause_fusion
from .reference_checkpoint import (
    build_reference_checkpoint,
    save_reference_checkpoint,
)
from .shift_sentinels import (
    ShiftSentinelModel,
    assess_support,
    fit_shift_sentinels,
)
from .temporal_episode_gate import (
    EpisodeRecord,
    GateState,
    SupportAssessment,
    TemporalEpisodeGate,
    TemporalGateConfig,
    ablation_configs,
    assemble_episode_records,
    calibrate_temporal_gate,
    default_temporal_config_grid,
    run_independent_window_policy,
    scenario_cluster_bootstrap,
    summarize_episode_results,
)


BENCHMARK_VERSION = "0.5.0"
PROTOCOL_VERSION = "1.0.0"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _augment_operational_metadata(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expose only deployment configuration from legacy hidden sidecars.

    v0.5 writes these fields directly into new manifests. The frozen v0.4
    dataset predates that schema, so this compatibility path reads only the
    scenario plan's operational environment and never reads ground-truth,
    fault, propagation, or label fields.
    """

    output = {}
    allowed = {
        "retention_days",
        "base_occupancy_fraction",
        "catchup_recording_channels",
        "average_bitrate_mbps",
        "shared_storage",
        "redundant_middleware",
    }
    for record in records:
        enriched = dict(record)
        missing = [name for name in allowed if name not in enriched]
        if missing:
            hidden = _load_json(Path(str(record["hidden_path"])))
            environment = hidden["scenario_plan"]["environment"]
            for name in allowed:
                enriched[name] = environment[name]
        output[str(record["scenario_id"])] = enriched
    return output


def _sequence_inputs(
    *,
    records: Sequence[EpisodeRecord],
    incident_model: IncidentDetectorModel,
) -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
]:
    probabilities = {}
    impact_signals = {}
    recovery_signals = {}
    for record in records:
        scores = tuple(
            incident_model.predict_score(window) for window in record.windows
        )
        probability_drops = tuple(
            0.0
            if index == 0
            else max(0.0, scores[index - 1] - scores[index])
            for index in range(len(scores))
        )
        impacts = []
        for window in record.windows:
            features = incident_features(window)
            impacts.append(
                max(
                    float(features["anomaly_fraction_positive"]),
                    float(features["error_fraction_positive"]),
                )
            )
        probabilities[record.scenario_id] = scores
        impact_signals[record.scenario_id] = tuple(impacts)
        recovery_signals[record.scenario_id] = probability_drops
    return probabilities, impact_signals, recovery_signals


def _support_inputs(
    *,
    records: Sequence[EpisodeRecord],
    model: ShiftSentinelModel,
    scenario_metadata: dict[str, dict[str, Any]],
) -> dict[str, tuple[SupportAssessment, ...]]:
    return {
        record.scenario_id: tuple(
            assess_support(
                model=model,
                window=window,
                metadata=scenario_metadata[record.scenario_id],
            )
            for window in record.windows
        )
        for record in records
    }


def _evaluate_policy(
    *,
    records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    impact_signals: dict[str, Sequence[float]],
    recovery_signals: dict[str, Sequence[float]],
    threshold: float,
    config: TemporalGateConfig | None,
    support: dict[str, Sequence[SupportAssessment]] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    results = []
    for record in records:
        assessments = (
            support[record.scenario_id] if support is not None else None
        )
        if config is None:
            result = run_independent_window_policy(
                record=record,
                probabilities=probabilities[record.scenario_id],
                threshold=threshold,
                support=assessments,
            )
        else:
            result = TemporalEpisodeGate(config).run(
                record=record,
                probabilities=probabilities[record.scenario_id],
                support=assessments,
                impact_signals=impact_signals[record.scenario_id],
                recovery_signals=recovery_signals[record.scenario_id],
            )
        results.append(result)
    return results, summarize_episode_results(results)


def _scenario_utility(result: Any) -> float:
    selected = [
        decision.state in {GateState.INCIDENT, GateState.ESCALATE}
        for decision in result.decisions
    ]
    if any(result.truth):
        return float(
            any(
                chosen and truth
                for chosen, truth in zip(selected, result.truth, strict=True)
            )
        )
    return float(not any(selected))


def _paired_bootstrap(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    *,
    seed: int,
) -> dict[str, float]:
    reference_by_id = {
        result.scenario_id: result for result in reference
    }
    differences_by_topology: dict[str, list[float]] = {}
    for result in candidate:
        differences_by_topology.setdefault(
            result.topology_group, []
        ).append(
            _scenario_utility(result)
            - _scenario_utility(reference_by_id[result.scenario_id])
        )
    values = {
        topology: sum(differences) / len(differences)
        for topology, differences in differences_by_topology.items()
    }
    return scenario_cluster_bootstrap(
        values_by_scenario=values,
        seed=seed,
    )


def _support_audit(
    *,
    records: Sequence[EpisodeRecord],
    support: dict[str, Sequence[SupportAssessment]],
    support_threshold: float,
) -> dict[str, Any]:
    by_split: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        by_split.setdefault(record.split, []).append(record)
    result = {}
    for split, split_records in by_split.items():
        unsupported_scenarios = [
            record
            for record in split_records
            if any(
                assessment.status.value == "unsupported"
                for assessment in support[record.scenario_id]
            )
        ]
        known_ood = [
            record for record in split_records if record.ood_axis_labels
        ]
        detected_known_ood = [
            record for record in known_ood if record in unsupported_scenarios
        ]
        axis_counts: dict[str, dict[str, int]] = {}
        for record in known_ood:
            for axis in record.ood_axis_labels:
                row = axis_counts.setdefault(
                    axis, {"scenarios": 0, "detected": 0}
                )
                row["scenarios"] += 1
                row["detected"] += int(
                    any(
                        dict(assessment.axis_scores).get(axis, 0.0)
                        > support_threshold
                        for assessment in support[record.scenario_id]
                    )
                )
        result[split] = {
            "scenario_count": len(split_records),
            "unsupported_scenario_count": len(unsupported_scenarios),
            "unsupported_scenario_rate": (
                len(unsupported_scenarios) / len(split_records)
                if split_records
                else 0.0
            ),
            "labeled_ood_scenario_count": len(known_ood),
            "labeled_ood_detection_rate": (
                len(detected_known_ood) / len(known_ood)
                if known_ood
                else 0.0
            ),
            "axis_detection": {
                axis: {
                    **row,
                    "rate": (
                        row["detected"] / row["scenarios"]
                        if row["scenarios"]
                        else 0.0
                    ),
                }
                for axis, row in axis_counts.items()
            },
        }
    return result


def _config_dict(config: TemporalGateConfig) -> dict[str, Any]:
    return asdict(config)


def _write_report(path: Path, results: dict[str, Any]) -> None:
    validation = results["ablations"]["G_full_rtieg"]["validation"]
    test = results["ablations"]["G_full_rtieg"]["test"]
    ood = results["ablations"]["G_full_rtieg"]["ood_test"]
    calibration = results["calibration"]
    acceptance = results["acceptance"]
    lines = [
        "# NRIM Scenario-Clustered Temporal Gate Freeze Report",
        "",
        f"- Benchmark version: `{results['benchmark_version']}`",
        f"- Release status: `{results['release_status']}`",
        (
            "- Dataset fingerprint: "
            f"`{results['dataset_fingerprint']}`"
        ),
        (
            "- Reference-model checkpoint: "
            f"`{results['reference_checkpoint_sha256']}`"
        ),
        "- Decision metric unit: `scenario`",
        "- Independent risk/bootstrap unit: `topology group`",
        "- Window overlap treated as dependent: `true`",
        "",
        "## Frozen policy",
        "",
        f"```json\n{json.dumps(results['selected_policy'], indent=2)}\n```",
        "",
        "## Scenario-level results",
        "",
        "| Split | Selection / escalation | Risk UCB | Episode recall | Median delay (h) | Fragmentation | Unsupported semantics |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in (
        ("validation", validation),
        ("test", test),
        ("ood_test", ood),
    ):
        delay = row["median_detection_delay_hours"]
        lines.append(
            f"| {split} "
            f"| {row['healthy_scenario_false_selection_rate']:.4f} "
            f"| {row['healthy_scenario_false_selection_ucb']:.4f} "
            f"| {row['incident_episode_recall']:.4f} "
            f"| {delay if delay is not None else 'n/a'} "
            f"| {row['mean_fragmentation']:.4f} "
            f"| {row['unsupported_safe_semantics_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Shift-support results",
            "",
            (
                "- OOD scenarios marked unsupported: "
                f"`{results['shift_sentinel']['audit']['ood_test']['labeled_ood_detection_rate']:.4f}`"
            ),
            (
                "- In-distribution validation scenarios marked unsupported: "
                f"`{results['shift_sentinel']['audit']['validation']['unsupported_scenario_rate']:.4f}`"
            ),
            (
                "- Unsupported OOD window decisions: "
                f"`UNKNOWN={ood['unsupported_unknown_rate']:.4f}`, "
                f"`ESCALATE={ood['unsupported_escalation_rate']:.4f}`"
            ),
            "",
            "## Registered acceptance checks",
            "",
        ]
    )
    for name, row in acceptance.items():
        lines.append(
            f"- {name}: `{'PASS' if row['passed'] else 'FAIL'}` - "
            f"{row['detail']}"
        )
    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            (
                "The selected point is "
                f"`{'feasible' if calibration['feasible'] else 'not feasible'}` "
                "under the registered finite-sample bounds. A zero observed "
                "false-episode count is not reported as zero risk; the exact "
                "one-sided binomial upper bound is used."
            ),
            "",
            "## OOD semantics",
            "",
            (
                "Unsupported low-score observations become `UNKNOWN`; "
                "unsupported high-score observations become `ESCALATE`. "
                "Neither is silently emitted as `HEALTHY`, and Stage 2 output "
                "on unsupported inputs is explicitly uncalibrated evidence."
            ),
            "",
            "## Stage 2",
            "",
            (
                "Learned-fusion ranking is frozen in the reference checkpoint. "
                "The temporal gate does not refit or alter Stage 2 scores."
            ),
            "",
            "## Limitations",
            "",
            "- The benchmark is entirely synthetic.",
            "- Exact group-risk bounds can remain inconclusive when a confounder cohort has few independent scenarios.",
            "- The shift sentinel detects support mismatch, not semantic novelty or guaranteed model failure.",
            "- Synthetic benchmark performance does not establish production performance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    nrim_dir = Path(__file__).resolve().parent.parent
    simulation_dir = nrim_dir / "examples" / "simulation"
    model_root = simulation_dir / "day09_model_ready_v2"
    source_root = simulation_dir / "day08_dataset_v2"
    previous_root = model_root / "benchmark_v0_4_0"
    output_root = model_root / "benchmark_v0_5_0"
    output_root.mkdir(parents=True, exist_ok=True)

    model_manifest = load_model_ready_manifest(
        model_root / "manifest.json"
    )
    source_manifest = _load_json(source_root / "manifest.json")
    previous_freeze = _load_json(previous_root / "benchmark_freeze.json")
    previous_diagnostics = _load_json(
        previous_root
        / "diagnostic_experiments"
        / "diagnostic_experiments.json"
    )
    scenario_metadata = _augment_operational_metadata(
        source_manifest["records"]
    )

    print("Loading immutable benchmark windows")
    windows_by_split = {
        split: load_split_windows(manifest=model_manifest, split=split)
        for split in ("train", "validation", "test", "ood_test")
    }
    all_windows = [
        window
        for split in ("train", "validation", "test", "ood_test")
        for window in windows_by_split[split]
    ]
    records, sequence_audit = assemble_episode_records(
        windows=all_windows,
        scenario_metadata=scenario_metadata,
    )
    records_by_split = {
        split: [record for record in records if record.split == split]
        for split in ("train", "validation", "test", "ood_test")
    }

    print("Reproducing and freezing Stage 1 v2 reference")
    incident_model = fit_incident_detector(
        training_windows=windows_by_split["train"],
        scenario_metadata=scenario_metadata,
        use_counterfactual_pairs=True,
        cohort_balancing=True,
    )
    threshold_selection = choose_incident_threshold(
        validation_windows=windows_by_split["validation"],
        model=incident_model,
        scenario_metadata=scenario_metadata,
        maximum_false_selection_rate=0.10,
        minimum_incident_coverage=0.70,
        maximum_cohort_false_selection_rate=0.15,
    )
    reference_threshold = float(
        previous_diagnostics["stage1"]["selection"]["threshold"]
    )
    reference_reproduced = (
        abs(threshold_selection.threshold - reference_threshold) < 1e-12
    )

    print("Freezing learned-fusion Stage 2 reference")
    fusion_model = fit_root_cause_fusion(
        training_windows=windows_by_split["train"],
        use_topology=True,
    )
    dataset_fingerprint = str(
        previous_freeze["fingerprint"]["benchmark_sha256"]
    )
    checkpoint = build_reference_checkpoint(
        incident_model=incident_model,
        fusion_model=fusion_model,
        incident_threshold=threshold_selection.threshold,
        dataset_fingerprint=dataset_fingerprint,
    )
    checkpoint_path = output_root / "reference_model_checkpoint.json"
    save_reference_checkpoint(
        checkpoint=checkpoint,
        path=checkpoint_path,
    )

    print("Computing causal window scores")
    probabilities, impact_signals, recovery_signals = _sequence_inputs(
        records=records,
        incident_model=incident_model,
    )

    print("Fitting train-only shift sentinel")
    shift_model = fit_shift_sentinels(
        training_records=records_by_split["train"],
        scenario_metadata=scenario_metadata,
        registered_operational_bounds={
            "workload__log_room_count": (
                math.log1p(50.0),
                math.log1p(250.0),
            ),
            "workload__retention_days": (
                math.log1p(2.0),
                math.log1p(14.0),
            ),
            "telemetry__scenario_missing_fraction": (
                0.0,
                0.15,
            ),
        },
    )
    support = _support_inputs(
        records=records,
        model=shift_model,
        scenario_metadata=scenario_metadata,
    )

    print("Selecting temporal policy on validation scenario clusters")
    calibration = calibrate_temporal_gate(
        validation_records=records_by_split["validation"],
        probabilities=probabilities,
        impact_signals=impact_signals,
        recovery_signals=recovery_signals,
        support=support,
        candidate_configs=default_temporal_config_grid(
            window_threshold=threshold_selection.threshold
        ),
        maximum_global_risk_ucb=0.10,
        maximum_confounder_risk_ucb=0.15,
        minimum_episode_recall=0.70,
        maximum_median_delay_hours=3.0,
        risk_confidence=0.95,
    )
    fixed_duration_config = TemporalGateConfig(
        window_threshold=threshold_selection.threshold,
        evidence_center=max(0.05, threshold_selection.threshold * 0.8),
        decay=0.75,
        entry_threshold=1.0,
        exit_threshold=0.2,
        minimum_suspect_windows=2,
        minimum_recovery_windows=2,
        severity_override_probability=max(
            0.95, threshold_selection.threshold
        ),
    )
    configurations = ablation_configs(
        calibrated=calibration.config,
    )
    configurations["B_hysteresis"] = replace(
        fixed_duration_config,
        decay=0.0,
        entry_threshold=0.0,
        exit_threshold=0.0,
        minimum_suspect_windows=1,
        minimum_recovery_windows=2,
        severity_override_probability=threshold_selection.threshold,
    )
    configurations["C_evidence_accumulator"] = replace(
        fixed_duration_config,
        minimum_suspect_windows=0,
        minimum_recovery_windows=1,
    )
    configurations["D_duration_states"] = fixed_duration_config

    print("Running registered A-G ablations")
    ablations: dict[str, dict[str, Any]] = {}
    raw_results: dict[str, dict[str, list[Any]]] = {}
    for name, config in configurations.items():
        ablations[name] = {}
        raw_results[name] = {}
        use_support = name in {"F_shift_sentinel", "G_full_rtieg"}
        for split in ("validation", "test", "ood_test"):
            split_results, metrics = _evaluate_policy(
                records=records_by_split[split],
                probabilities=probabilities,
                impact_signals=impact_signals,
                recovery_signals=recovery_signals,
                threshold=threshold_selection.threshold,
                config=config,
                support=support if use_support else None,
            )
            raw_results[name][split] = split_results
            ablations[name][split] = metrics

    paired_bootstrap = {
        name: {
            split: _paired_bootstrap(
                raw_results["A_independent_window"][split],
                raw_results[name][split],
                seed=5_000 + 100 * ablation_index + split_index,
            )
            for split_index, split in enumerate(
                ("validation", "test", "ood_test")
            )
        }
        for ablation_index, name in enumerate(configurations)
    }
    support_audit = _support_audit(
        records=records,
        support=support,
        support_threshold=shift_model.support_threshold,
    )

    validation = ablations["G_full_rtieg"]["validation"]
    test = ablations["G_full_rtieg"]["test"]
    ood = ablations["G_full_rtieg"]["ood_test"]
    validation_reference = ablations["A_independent_window"]["validation"]
    previous_findings_clean = not previous_freeze.get("findings")
    acceptance = {
        "reference_reproduction": {
            "passed": reference_reproduced,
            "detail": (
                f"threshold={threshold_selection.threshold:.12f}, "
                f"reference={reference_threshold:.12f}"
            ),
        },
        "validation_global_risk_ucb": {
            "passed": (
                validation["healthy_scenario_false_selection_ucb"] <= 0.10
            ),
            "detail": (
                f"UCB={validation['healthy_scenario_false_selection_ucb']:.4f} "
                "<= 0.10"
            ),
        },
        "validation_confounder_risk_ucb": {
            "passed": validation["worst_confounder_ucb"] <= 0.15,
            "detail": (
                f"worst UCB={validation['worst_confounder_ucb']:.4f} "
                "<= 0.15"
            ),
        },
        "validation_episode_recall": {
            "passed": validation["incident_episode_recall"] >= 0.70,
            "detail": (
                f"recall={validation['incident_episode_recall']:.4f} >= 0.70"
            ),
        },
        "test_point_risk": {
            "passed": (
                test["healthy_scenario_false_selection_rate"] <= 0.12
            ),
            "detail": (
                f"risk={test['healthy_scenario_false_selection_rate']:.4f} "
                "<= 0.12"
            ),
        },
        "test_episode_recall": {
            "passed": test["incident_episode_recall"] >= 0.70,
            "detail": (
                f"recall={test['incident_episode_recall']:.4f} >= 0.70"
            ),
        },
        "ood_safe_semantics": {
            "passed": ood["unsupported_safe_semantics_rate"] == 1.0,
            "detail": (
                "all unsupported decisions are UNKNOWN or ESCALATE"
            ),
        },
        "temporal_stability": {
            "passed": (
                validation[
                    "false_episodes_per_100_healthy_scenario_hours"
                ]
                < validation_reference[
                    "false_episodes_per_100_healthy_scenario_hours"
                ]
                and validation["mean_fragmentation"]
                <= validation_reference["mean_fragmentation"]
                and (
                    validation["median_detection_delay_hours"] is not None
                    and validation["median_detection_delay_hours"] <= 3.0
                )
            ),
            "detail": (
                "fewer healthy false episodes, no added fragmentation, "
                "and median delay <= 3 hours"
            ),
        },
        "stage2_preserved": {
            "passed": checkpoint["ranking_policy"][
                "modified_by_temporal_gate"
            ]
            is False,
            "detail": "learned-fusion weights frozen in checkpoint",
        },
        "split_and_shortcut_audits": {
            "passed": sequence_audit.leakage_free
            and previous_findings_clean,
            "detail": (
                f"sequence_leakage_free={sequence_audit.leakage_free}, "
                f"prior_findings_clean={previous_findings_clean}"
            ),
        },
    }
    release_status = (
        "frozen"
        if all(row["passed"] for row in acceptance.values())
        else "candidate"
    )
    results = {
        "benchmark_name": (
            "NRIM Scenario-Clustered Risk-Controlled Temporal Episode Gate"
        ),
        "benchmark_version": BENCHMARK_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": release_status,
        "dataset_fingerprint": dataset_fingerprint,
        "reference_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "reference_stage1": {
            "threshold": threshold_selection.threshold,
            "window_level_feasible": threshold_selection.feasible,
            "reference_reproduced": reference_reproduced,
        },
        "stage2_reference": {
            "model": "learned_fusion",
            "topology_retained": True,
            "checkpoint_frozen": True,
            "temporal_gate_changes_ranking": False,
        },
        "sequence_audit": asdict(sequence_audit),
        "risk_assumptions": {
            "decision_metric_unit": "scenario",
            "independent_risk_unit": "topology_group",
            "bootstrap_unit": "topology_group",
            "confidence": 0.95,
            "risk_bound": "one_sided_exact_clopper_pearson",
            "overlapping_windows_independent": False,
            "calibration_split": "validation",
            "test_used_for_selection": False,
        },
        "calibration": {
            "feasible": calibration.feasible,
            "objective": calibration.objective,
            "evaluated_configurations": (
                calibration.evaluated_configurations
            ),
            "feasibility_failures": calibration.feasibility_failures,
            "validation_metrics_without_shift_override": calibration.metrics,
        },
        "selected_policy": _config_dict(calibration.config),
        "shift_sentinel": {
            "fit_split": "train",
            "axes": [
                "topology",
                "workload",
                "telemetry",
                "applicability",
                "domain",
            ],
            "synthetic_shift_calibration_only": True,
            "registered_id_support": {
                "room_count": [50, 250],
                "retention_days": [2, 14],
                "scenario_missing_fraction": [0.0, 0.15],
            },
            "operational_context_is_not_model_label": True,
            "legacy_context_source": (
                "scenario_plan.environment only; future manifests "
                "store these fields directly"
            ),
            "support_threshold": shift_model.support_threshold,
            "audit": support_audit,
        },
        "ablations": ablations,
        "paired_topology_bootstrap_vs_independent_window": (
            paired_bootstrap
        ),
        "acceptance": acceptance,
        "limitations": [
            "The benchmark is entirely synthetic.",
            (
                "Small independent scenario cohorts can make exact "
                "finite-sample risk bounds inconclusive."
            ),
            (
                "Support detection cannot guarantee semantic OOD "
                "detection or safe prediction."
            ),
            (
                "Synthetic benchmark performance does not establish "
                "production performance."
            ),
        ],
    }
    json_path = output_root / "temporal_episode_experiment.json"
    report_path = output_root / "freeze_report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(report_path, results)
    print("Saved:", checkpoint_path)
    print("Saved:", json_path)
    print("Saved:", report_path)
    print("Release status:", release_status)


if __name__ == "__main__":
    main()
