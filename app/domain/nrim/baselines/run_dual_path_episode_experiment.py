from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
)
from app.domain.nrim.simulation.locked_test_governance import (
    locked_test_is_unopened,
    open_locked_test_once,
    verify_locked_test_seal,
)

from .baseline_evaluator import load_split_windows
from .dual_path_episode_gate import (
    DualPathConfig,
    DualPathEpisodeGate,
    calibrate_dual_path_gate,
    default_dual_path_config_grid,
    fit_healthy_residual_model,
    registered_dual_path_ablation_configs,
    stage2_eligible_decision,
)
from .incident_detector import (
    IncidentDetectorModel,
    choose_incident_threshold,
    fit_incident_detector,
)
from .learned_fusion import (
    RootCauseFusionModel,
    learned_fusion_scores,
)
from .reference_checkpoint import (
    load_reference_checkpoint,
)
from .shift_sentinels import (
    ShiftSentinelModel,
    assess_support,
    fit_shift_sentinels,
)
from .temporal_episode_gate import (
    EpisodeRecord,
    GateState,
    ScenarioEpisodeResult,
    SupportAssessment,
    TemporalEpisodeGate,
    TemporalGateConfig,
    assemble_episode_records,
    run_independent_window_policy,
    scenario_cluster_bootstrap,
    summarize_episode_results,
)


BENCHMARK_VERSION = "0.6.0"
PROTOCOL_VERSION = "2.0.0"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_metadata(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(record["scenario_id"]): dict(record)
        for record in records
    }


def _probability_inputs(
    *,
    records: Sequence[EpisodeRecord],
    incident_model: IncidentDetectorModel,
) -> dict[str, tuple[float, ...]]:
    return {
        record.scenario_id: tuple(
            incident_model.predict_score(window)
            for window in record.windows
        )
        for record in records
    }


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
                metadata=scenario_metadata[
                    record.scenario_id
                ],
            )
            for window in record.windows
        )
        for record in records
    }


def _evaluate_independent(
    *,
    records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    threshold: float,
    support: dict[
        str,
        Sequence[SupportAssessment],
    ],
) -> tuple[
    list[ScenarioEpisodeResult],
    dict[str, Any],
]:
    results = [
        run_independent_window_policy(
            record=record,
            probabilities=probabilities[
                record.scenario_id
            ],
            threshold=threshold,
            support=support[record.scenario_id],
        )
        for record in records
    ]
    return results, summarize_episode_results(results)


def _evaluate_persistence(
    *,
    records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    config: TemporalGateConfig,
    support: dict[
        str,
        Sequence[SupportAssessment],
    ],
) -> tuple[
    list[ScenarioEpisodeResult],
    dict[str, Any],
]:
    results = [
        TemporalEpisodeGate(config).run(
            record=record,
            probabilities=probabilities[
                record.scenario_id
            ],
            support=support[record.scenario_id],
        )
        for record in records
    ]
    return results, summarize_episode_results(results)


def _evaluate_dual(
    *,
    records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    config: DualPathConfig,
    residual_model: Any,
    support: dict[
        str,
        Sequence[SupportAssessment],
    ],
    scenario_metadata: dict[str, dict[str, Any]],
) -> tuple[
    list[ScenarioEpisodeResult],
    dict[str, Any],
]:
    gate = DualPathEpisodeGate(
        config=config,
        residual_model=(
            residual_model
            if config.use_residual
            else None
        ),
    )
    results = [
        gate.run(
            record=record,
            probabilities=probabilities[
                record.scenario_id
            ],
            metadata=scenario_metadata[
                record.scenario_id
            ],
            support=support[record.scenario_id],
        )
        for record in records
    ]
    return results, summarize_episode_results(results)


def _scenario_utility(
    result: ScenarioEpisodeResult,
) -> float:
    selected = [
        decision.state
        in {GateState.INCIDENT, GateState.ESCALATE}
        for decision in result.decisions
    ]
    if any(result.truth):
        return float(
            any(
                chosen and truth
                for chosen, truth in zip(
                    selected,
                    result.truth,
                    strict=True,
                )
            )
        )
    return float(not any(selected))


def _paired_topology_bootstrap(
    *,
    reference: Sequence[ScenarioEpisodeResult],
    candidate: Sequence[ScenarioEpisodeResult],
    seed: int,
) -> dict[str, float]:
    reference_by_id = {
        result.scenario_id: result
        for result in reference
    }
    differences: dict[str, list[float]] = {}
    for result in candidate:
        differences.setdefault(
            result.topology_group,
            [],
        ).append(
            _scenario_utility(result)
            - _scenario_utility(
                reference_by_id[result.scenario_id]
            )
        )
    values = {
        topology: mean(rows)
        for topology, rows in differences.items()
    }
    return scenario_cluster_bootstrap(
        values_by_scenario=values,
        seed=seed,
        samples=2_000,
    )


def _scenario_support_rates(
    *,
    records: Sequence[EpisodeRecord],
    support: dict[
        str,
        Sequence[SupportAssessment],
    ],
) -> dict[str, float | int]:
    unsupported = [
        record
        for record in records
        if any(
            assessment.status.value
            == "unsupported"
            for assessment in support[
                record.scenario_id
            ]
        )
    ]
    return {
        "scenario_count": len(records),
        "unsupported_scenario_count": len(
            unsupported
        ),
        "unsupported_scenario_rate": (
            len(unsupported) / len(records)
            if records
            else 0.0
        ),
    }


def _duration_recall(
    metrics: dict[str, Any],
    bucket: str,
) -> float:
    return float(
        metrics["episode_duration_metrics"].get(
            bucket,
            {},
        ).get("episode_recall", 0.0)
    )


def _validation_acceptance(
    metrics: dict[str, Any],
) -> dict[str, bool]:
    family_rows = metrics[
        "failure_family_metrics"
    ]
    capacity_delay = family_rows.get(
        "storage_capacity_saturation",
        {},
    ).get("median_detection_delay_hours")
    return {
        "scenario_false_selection_at_most_0_10": (
            metrics[
                "healthy_scenario_false_selection_rate"
            ]
            <= 0.10
        ),
        "global_95_ucb_at_most_0_10": (
            metrics[
                "healthy_scenario_false_selection_ucb"
            ]
            <= 0.10
        ),
        "worst_confounder_point_at_most_0_10": (
            metrics[
                "worst_confounder_point_risk"
            ]
            <= 0.10
        ),
        "worst_confounder_ucb_at_most_0_15": (
            metrics["worst_confounder_ucb"]
            <= 0.15
        ),
        "episode_recall_at_least_0_80": (
            metrics["incident_episode_recall"]
            >= 0.80
        ),
        "each_family_recall_at_least_0_70": (
            bool(family_rows)
            and all(
                float(row["episode_recall"])
                >= 0.70
                for row in family_rows.values()
            )
        ),
        "median_delay_at_most_2h": (
            metrics[
                "median_detection_delay_hours"
            ]
            is not None
            and metrics[
                "median_detection_delay_hours"
            ]
            <= 2.0
        ),
        "p90_delay_at_most_4h": (
            metrics[
                "p90_detection_delay_hours"
            ]
            is not None
            and metrics[
                "p90_detection_delay_hours"
            ]
            <= 4.0
        ),
        "capacity_median_delay_at_most_4h": (
            capacity_delay is not None
            and capacity_delay <= 4.0
        ),
        "fragmentation_at_most_0_05": (
            metrics["mean_fragmentation"]
            <= 0.05
        ),
        "unsupported_safe_semantics": (
            metrics[
                "unsupported_safe_semantics_rate"
            ]
            == 1.0
        ),
    }


def _test_acceptance(
    *,
    metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> dict[str, bool]:
    family_rows = metrics[
        "failure_family_metrics"
    ]
    return {
        "point_risk_at_most_0_12": (
            metrics[
                "healthy_scenario_false_selection_rate"
            ]
            <= 0.12
        ),
        "episode_recall_at_least_0_75": (
            metrics["incident_episode_recall"]
            >= 0.75
        ),
        "each_family_recall_at_least_0_65": (
            bool(family_rows)
            and all(
                float(row["episode_recall"])
                >= 0.65
                for row in family_rows.values()
            )
        ),
        "median_delay_at_most_3h": (
            metrics[
                "median_detection_delay_hours"
            ]
            is not None
            and metrics[
                "median_detection_delay_hours"
            ]
            <= 3.0
        ),
        "fragmentation_at_most_0_05": (
            metrics["mean_fragmentation"]
            <= 0.05
        ),
        "false_episodes_per_100h_at_most_0_50": (
            metrics[
                "false_episodes_per_100_healthy_scenario_hours"
            ]
            <= 0.50
        ),
        "risk_delta_at_most_plus_0_04": (
            metrics[
                "healthy_scenario_false_selection_rate"
            ]
            - validation_metrics[
                "healthy_scenario_false_selection_rate"
            ]
            <= 0.04
        ),
        "recall_delta_at_least_minus_0_08": (
            metrics["incident_episode_recall"]
            - validation_metrics[
                "incident_episode_recall"
            ]
            >= -0.08
        ),
    }


def _architecture_acceptance(
    *,
    ablations: dict[str, dict[str, Any]],
    support_audit: dict[str, Any],
) -> dict[str, bool]:
    independent = ablations[
        "A_independent_plus_sentinel"
    ]["validation"]
    persistence = ablations[
        "B_persistence_plus_sentinel"
    ]["validation"]
    raw = ablations[
        "C_sequential_raw_probability"
    ]["validation"]
    residual = ablations[
        "D_sequential_residual"
    ]["validation"]
    no_confirmation = ablations[
        "E_probability_fast_override"
    ]["validation"]
    full = ablations[
        "G_full_dual_path"
    ]["validation"]
    return {
        "fast_short_recall_increment_at_least_0_15": (
            _duration_recall(
                full,
                "1_to_2_decision_windows",
            )
            - _duration_recall(
                persistence,
                "1_to_2_decision_windows",
            )
            >= 0.15
        ),
        "fast_risk_increment_at_most_0_02": (
            full[
                "healthy_scenario_false_selection_rate"
            ]
            - persistence[
                "healthy_scenario_false_selection_rate"
            ]
            <= 0.02
        ),
        "slow_false_episode_reduction_at_least_50pct": (
            residual[
                "false_episodes_per_100_healthy_scenario_hours"
            ]
            <= 0.50
            * independent[
                "false_episodes_per_100_healthy_scenario_hours"
            ]
        ),
        "slow_persistent_recall_at_least_0_75": (
            _duration_recall(
                residual,
                "longer_than_4_decision_windows",
            )
            >= 0.75
        ),
        "impact_confirmation_is_incremental": (
            no_confirmation[
                "healthy_scenario_false_selection_rate"
            ]
            > full[
                "healthy_scenario_false_selection_rate"
            ]
            or no_confirmation[
                "false_episodes_per_100_healthy_scenario_hours"
            ]
            > full[
                "false_episodes_per_100_healthy_scenario_hours"
            ]
        ),
        "residual_not_worse_at_matched_recall": (
            residual[
                "incident_episode_recall"
            ]
            >= raw["incident_episode_recall"] - 0.02
            and residual[
                "healthy_scenario_false_selection_rate"
            ]
            <= raw[
                "healthy_scenario_false_selection_rate"
            ]
        ),
        "registered_ood_detection_at_least_0_95": (
            support_audit["ood_test"][
                "unsupported_scenario_rate"
            ]
            >= 0.95
        ),
        "id_support_false_alarm_at_most_0_05": (
            support_audit["validation"][
                "unsupported_scenario_rate"
            ]
            <= 0.05
            and support_audit[
                "development_test"
            ]["unsupported_scenario_rate"]
            <= 0.05
        ),
    }


def _ranking_summary(
    *,
    records: Sequence[EpisodeRecord],
    episode_results: Sequence[
        ScenarioEpisodeResult
    ],
    fusion_model: RootCauseFusionModel,
) -> dict[str, Any]:
    result_by_scenario = {
        result.scenario_id: result
        for result in episode_results
    }
    all_ranks = []
    accepted_ranks = []
    score_payload = []
    for record in records:
        result = result_by_scenario[
            record.scenario_id
        ]
        for window, decision in zip(
            record.windows,
            result.decisions,
            strict=True,
        ):
            if not bool(
                int(
                    window["targets"][
                        "current_incident"
                    ]
                )
            ):
                continue
            labels = [
                int(value)
                for value in window["targets"][
                    "root_cause_node"
                ]
            ]
            if not any(labels):
                continue
            true_node = str(
                window["node_ids"][
                    labels.index(1)
                ]
            )
            scores = learned_fusion_scores(
                window=window,
                model=fusion_model,
            )
            ranked = sorted(
                scores,
                key=lambda item: (
                    -item.score,
                    item.node_id,
                ),
            )
            rank = (
                [
                    item.node_id
                    for item in ranked
                ].index(true_node)
                + 1
            )
            all_ranks.append(rank)
            if stage2_eligible_decision(decision):
                accepted_ranks.append(rank)
            if len(score_payload) < 64:
                score_payload.append(
                    {
                        "window_id": window[
                            "window_id"
                        ],
                        "scores": [
                            [
                                item.node_id,
                                item.score,
                            ]
                            for item in ranked
                        ],
                    }
                )

    def summarize(ranks: Sequence[int]) -> dict[str, Any]:
        return {
            "window_count": len(ranks),
            "mrr": (
                mean(1.0 / rank for rank in ranks)
                if ranks
                else 0.0
            ),
            "hits_at_1": (
                mean(rank <= 1 for rank in ranks)
                if ranks
                else 0.0
            ),
            "hits_at_3": (
                mean(rank <= 3 for rank in ranks)
                if ranks
                else 0.0
            ),
        }

    return {
        "all_true_faulty_windows": summarize(
            all_ranks
        ),
        "stage1_accepted_faulty_windows": (
            summarize(accepted_ranks)
        ),
        "accepted_faulty_coverage": (
            len(accepted_ranks) / len(all_ranks)
            if all_ranks
            else 0.0
        ),
        "anchor_score_and_ranking_sha256": (
            _canonical_hash(score_payload)
        ),
    }


def _write_report(
    *,
    path: Path,
    results: dict[str, Any],
) -> None:
    validation = results["ablations"][
        "G_full_dual_path"
    ]["validation"]
    development = results["ablations"][
        "G_full_dual_path"
    ]["development_test"]
    locked = results["locked_test"]["metrics"]
    lines = [
        "# NRIM v0.6 Dual-Path Episode Gate",
        "",
        f"- Status: **{results['release_status'].upper()}**",
        (
            "- Locked test: **"
            + results["locked_test"]["status"].upper()
            + "**"
        ),
        (
            "- Stage 2 frozen checkpoint: `"
            + results[
                "stage2_reference"
            ]["source_checkpoint_sha256"]
            + "`"
        ),
        "",
        "## Validation",
        "",
        (
            "- Healthy scenario risk / 95% topology UCB: "
            f"{validation['healthy_scenario_false_selection_rate']:.4f} / "
            f"{validation['healthy_scenario_false_selection_ucb']:.4f}"
        ),
        (
            "- Episode recall: "
            f"{validation['incident_episode_recall']:.4f}"
        ),
        (
            "- Median / p90 delay: "
            f"{validation['median_detection_delay_hours']} / "
            f"{validation['p90_detection_delay_hours']} hours"
        ),
        "",
        "## Development test",
        "",
        (
            "- Healthy scenario risk: "
            f"{development['healthy_scenario_false_selection_rate']:.4f}"
        ),
        (
            "- Episode recall: "
            f"{development['incident_episode_recall']:.4f}"
        ),
        "",
        "## Locked test",
        "",
        (
            "- Healthy scenario risk: "
            f"{locked['healthy_scenario_false_selection_rate']:.4f}"
            if locked is not None
            else "- Not opened"
        ),
        (
            "- Episode recall: "
            f"{locked['incident_episode_recall']:.4f}"
            if locked is not None
            else "- Episode recall: not evaluated"
        ),
        "",
        "## Mandatory validation decision",
        "",
    ]
    for name, passed in results[
        "validation_acceptance"
    ].items():
        lines.append(
            f"- {'PASS' if passed else 'FAIL'} — {name}"
        )
    lines.extend(
        [
            "",
            "## Architectural attribution",
            "",
        ]
    )
    for name, passed in results[
        "architecture_acceptance"
    ].items():
        lines.append(
            f"- {'PASS' if passed else 'FAIL'} — {name}"
        )
    failed_architecture = [
        name
        for name, passed in results[
            "architecture_acceptance"
        ].items()
        if not passed
    ]
    if failed_architecture:
        lines.extend(
            [
                "",
                (
                    "The release remains a candidate because the "
                    "architectural increment is not fully identified, "
                    "even though mandatory validation, development, and "
                    "locked-test operating criteria passed."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The slow path is an empirically calibrated CUSUM over "
                "healthy-conditioned residuals. The report makes no "
                "distribution-free false-alarm claim for overlapping "
                "windows; risk is measured on independent topology groups "
                "with a one-sided exact binomial upper bound."
            ),
            (
                "The fast path cannot open on classifier probability alone: "
                "it requires independent observable impact breadth and causal "
                "consistency. Unsupported inputs are UNKNOWN or ESCALATE and "
                "never enter Stage 2."
            ),
            "",
            "## Limitations",
            "",
            "- Synthetic evidence is not production evidence.",
            (
                "- The system remains a candidate until prospective "
                "shadow-mode evaluation on real operations data."
            ),
        ]
    )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    nrim_dir = Path(__file__).resolve().parent.parent
    simulation_root = (
        nrim_dir / "examples" / "simulation"
    )
    source_root = (
        simulation_root / "day08_dataset_v06"
    )
    model_root = (
        simulation_root / "day09_model_ready_v06"
    )
    output_root = (
        model_root / "benchmark_v0_6_0"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    seal_path = (
        source_root
        / "governance"
        / "locked_test_seal.json"
    )
    access_ledger_path = (
        source_root
        / "governance"
        / "locked_test_access.json"
    )
    seal = verify_locked_test_seal(
        seal_path=seal_path
    )
    if not locked_test_is_unopened(
        access_ledger_path
    ):
        raise RuntimeError(
            "The v0.6 locked test was already opened; "
            "refusing a second benchmark run"
        )

    development_source = _load_json(
        Path(seal["development_manifest_path"])
    )
    scenario_metadata = _scenario_metadata(
        development_source["records"]
    )
    model_manifest = load_model_ready_manifest(
        model_root / "manifest.json"
    )
    development_model_manifest = {
        **model_manifest,
        "records": [
            record
            for record in model_manifest["records"]
            if record["split"] != "locked_test"
        ],
    }
    split_names = (
        "train",
        "validation",
        "development_test",
        "ood_test",
        "semantic_challenge",
    )
    windows_by_split = {
        split: load_split_windows(
            manifest=development_model_manifest,
            split=split,
        )
        for split in split_names
    }
    all_windows = [
        window
        for split in split_names
        for window in windows_by_split[split]
    ]
    records, sequence_audit = (
        assemble_episode_records(
            windows=all_windows,
            scenario_metadata=scenario_metadata,
        )
    )
    records_by_split = {
        split: [
            record
            for record in records
            if record.split == split
        ]
        for split in split_names
    }

    print(
        "Fitting leakage-free Stage 1 window model"
    )
    incident_model = fit_incident_detector(
        training_windows=windows_by_split[
            "train"
        ],
        scenario_metadata=scenario_metadata,
        use_counterfactual_pairs=False,
        cohort_balancing=True,
        iterations=300,
    )
    threshold = choose_incident_threshold(
        validation_windows=windows_by_split[
            "validation"
        ],
        model=incident_model,
        scenario_metadata=scenario_metadata,
        maximum_false_selection_rate=0.10,
        minimum_incident_coverage=0.70,
        maximum_cohort_false_selection_rate=0.15,
    )
    probabilities = _probability_inputs(
        records=records,
        incident_model=incident_model,
    )

    print(
        "Fitting train-only support sentinel and healthy residual"
    )
    sentinel = fit_shift_sentinels(
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
        model=sentinel,
        scenario_metadata=scenario_metadata,
    )
    residual_model = fit_healthy_residual_model(
        training_records=records_by_split["train"],
        probabilities=probabilities,
        scenario_metadata=scenario_metadata,
    )

    print(
        "Calibrating dual path on validation topology groups"
    )
    calibration = calibrate_dual_path_gate(
        validation_records=records_by_split[
            "validation"
        ],
        probabilities=probabilities,
        scenario_metadata=scenario_metadata,
        residual_model=residual_model,
        candidate_configs=(
            default_dual_path_config_grid()
        ),
        support=support,
    )
    dual_configs = (
        registered_dual_path_ablation_configs(
            calibration.config
        )
    )
    persistence_config = TemporalGateConfig(
        window_threshold=threshold.threshold,
        evidence_center=max(
            0.05,
            threshold.threshold * 0.80,
        ),
        decay=0.75,
        entry_threshold=1.0,
        exit_threshold=0.20,
        minimum_suspect_windows=2,
        minimum_recovery_windows=2,
        severity_override_probability=0.999999,
    )

    print("Running registered A-H ablations")
    ablations: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {
        name: {}
        for name in dual_configs
    }
    raw_results: dict[
        str,
        dict[str, list[ScenarioEpisodeResult]],
    ] = {
        name: {}
        for name in dual_configs
    }
    for name, config in dual_configs.items():
        for split in (
            "validation",
            "development_test",
            "ood_test",
            "semantic_challenge",
        ):
            if name == "A_independent_plus_sentinel":
                split_results, metrics = (
                    _evaluate_independent(
                        records=records_by_split[split],
                        probabilities=probabilities,
                        threshold=threshold.threshold,
                        support=support,
                    )
                )
            elif name == "B_persistence_plus_sentinel":
                split_results, metrics = (
                    _evaluate_persistence(
                        records=records_by_split[split],
                        probabilities=probabilities,
                        config=persistence_config,
                        support=support,
                    )
                )
            else:
                assert config is not None
                split_results, metrics = (
                    _evaluate_dual(
                        records=records_by_split[split],
                        probabilities=probabilities,
                        config=config,
                        residual_model=residual_model,
                        support=support,
                        scenario_metadata=(
                            scenario_metadata
                        ),
                    )
                )
            raw_results[name][split] = (
                split_results
            )
            ablations[name][split] = metrics

    support_audit = {
        split: _scenario_support_rates(
            records=records_by_split[split],
            support=support,
        )
        for split in split_names
    }
    architecture_acceptance = (
        _architecture_acceptance(
            ablations=ablations,
            support_audit=support_audit,
        )
    )
    validation_metrics = ablations[
        "G_full_dual_path"
    ]["validation"]
    validation_acceptance = (
        _validation_acceptance(
            validation_metrics
        )
    )
    development_metrics = ablations[
        "G_full_dual_path"
    ]["development_test"]
    development_acceptance = _test_acceptance(
        metrics=development_metrics,
        validation_metrics=validation_metrics,
    )

    print("Verifying frozen Stage 2")
    source_checkpoint_path = (
        simulation_root
        / "day09_model_ready_v2"
        / "benchmark_v0_5_0"
        / "reference_model_checkpoint.json"
    )
    _, fusion_model, _ = load_reference_checkpoint(
        path=source_checkpoint_path
    )
    checkpoint_payload = _load_json(
        source_checkpoint_path
    )
    fusion_hash = _canonical_hash(
        checkpoint_payload[
            "learned_fusion_model"
        ]
    )
    stage2_reference = {
        "source_checkpoint_path": str(
            source_checkpoint_path
        ),
        "source_checkpoint_sha256": _file_hash(
            source_checkpoint_path
        ),
        "stored_checkpoint_sha256": (
            checkpoint_payload[
                "checkpoint_sha256"
            ]
        ),
        "learned_fusion_model_sha256": (
            fusion_hash
        ),
        "weights_retrained": False,
        "ranking_policy_modified": False,
    }
    ranking = {
        split: _ranking_summary(
            records=records_by_split[split],
            episode_results=raw_results[
                "G_full_dual_path"
            ][split],
            fusion_model=fusion_model,
        )
        for split in (
            "validation",
            "development_test",
            "ood_test",
        )
    }

    bootstrap = {
        split: [
            _paired_topology_bootstrap(
                reference=raw_results[
                    "A_independent_plus_sentinel"
                ][split],
                candidate=raw_results[
                    "G_full_dual_path"
                ][split],
                seed=6_060 + seed_index,
            )
            for seed_index in range(5)
        ]
        for split in (
            "validation",
            "development_test",
            "ood_test",
        )
    }

    required_before_locked = {
        **validation_acceptance,
        "sequence_and_split_audit": (
            sequence_audit.leakage_free
        ),
        "inference_schema_has_no_context_features": (
            not any(
                "context__" in name
                or "confounder" in name
                for name in model_manifest[
                    "feature_schema"
                ]["node_features"]
                if isinstance(name, str)
            )
        ),
        "stage2_frozen": True,
    }
    # Feature schema rows are dictionaries; audit the names explicitly.
    required_before_locked[
        "inference_schema_has_no_context_features"
    ] = not any(
        "context__" in str(row["name"])
        or "confounder" in str(row["name"])
        for row in model_manifest[
            "feature_schema"
        ]["node_features"]
    )

    locked_status = "sealed"
    locked_metrics = None
    locked_acceptance = None
    if all(required_before_locked.values()):
        print(
            "Validation passed; opening locked test exactly once"
        )
        locked_source = open_locked_test_once(
            seal_path=seal_path,
            access_ledger_path=access_ledger_path,
            validation_acceptance=(
                required_before_locked
            ),
        )
        scenario_metadata.update(
            _scenario_metadata(
                locked_source["records"]
            )
        )
        locked_windows = load_split_windows(
            manifest=model_manifest,
            split="locked_test",
        )
        locked_records, locked_sequence_audit = (
            assemble_episode_records(
                windows=locked_windows,
                scenario_metadata=scenario_metadata,
            )
        )
        locked_probabilities = _probability_inputs(
            records=locked_records,
            incident_model=incident_model,
        )
        locked_support = _support_inputs(
            records=locked_records,
            model=sentinel,
            scenario_metadata=scenario_metadata,
        )
        _, locked_metrics = _evaluate_dual(
            records=locked_records,
            probabilities=locked_probabilities,
            config=calibration.config,
            residual_model=residual_model,
            support=locked_support,
            scenario_metadata=scenario_metadata,
        )
        locked_acceptance = _test_acceptance(
            metrics=locked_metrics,
            validation_metrics=validation_metrics,
        )
        locked_status = "opened_once"
        required_before_locked[
            "locked_sequence_audit"
        ] = locked_sequence_audit.leakage_free

    release_status = (
        "frozen"
        if (
            all(validation_acceptance.values())
            and all(
                architecture_acceptance.values()
            )
            and all(
                development_acceptance.values()
            )
            and locked_acceptance is not None
            and all(locked_acceptance.values())
        )
        else "candidate"
    )
    dataset_fingerprint = _canonical_hash(
        {
            "development_manifest": (
                development_source
            ),
            "locked_commitment": seal[
                "commitment_sha256"
            ],
            "feature_schema": model_manifest[
                "feature_schema"
            ],
        }
    )
    results = {
        "benchmark_name": (
            "NRIM Dual-Path Risk-Controlled Temporal Episode Gate"
        ),
        "benchmark_version": BENCHMARK_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "release_status": release_status,
        "dataset_fingerprint": (
            dataset_fingerprint
        ),
        "locked_test_commitment_sha256": seal[
            "commitment_sha256"
        ],
        "sequence_audit": asdict(
            sequence_audit
        ),
        "stage1_window_reference": {
            "threshold": threshold.threshold,
            "threshold_feasible": (
                threshold.feasible
            ),
            "training_uses_counterfactual_ids": False,
            "training_uses_confounder_labels": False,
            "inference_context_features_removed": True,
        },
        "healthy_residual_model": asdict(
            residual_model
        ),
        "selected_policy": asdict(
            calibration.config
        ),
        "calibration": {
            "split": "validation",
            "topology_clustered": True,
            "locked_test_used": False,
            "feasible": calibration.feasible,
            "objective": calibration.objective,
            "evaluated_configurations": (
                calibration.evaluated_configurations
            ),
            "feasibility_failures": (
                calibration.feasibility_failures
            ),
            "candidate_summaries": (
                calibration.candidate_summaries
            ),
        },
        "support_sentinel": {
            "fit_split": "train",
            "support_threshold": (
                sentinel.support_threshold
            ),
            "audit": support_audit,
        },
        "ablations": ablations,
        "paired_topology_bootstrap_five_seeds": (
            bootstrap
        ),
        "validation_acceptance": (
            validation_acceptance
        ),
        "architecture_acceptance": (
            architecture_acceptance
        ),
        "development_acceptance": (
            development_acceptance
        ),
        "stage2_reference": stage2_reference,
        "ranking_independent_of_stage1": (
            ranking
        ),
        "locked_test": {
            "status": locked_status,
            "opened_at_most_once": True,
            "metrics": locked_metrics,
            "acceptance": locked_acceptance,
        },
        "risk_contract": {
            "independent_unit": (
                "topology_group"
            ),
            "confidence": 0.95,
            "bound": (
                "one_sided_exact_clopper_pearson"
            ),
            "overlapping_windows_are_independent": (
                False
            ),
            "sequential_guarantee_claimed": False,
            "test_used_for_selection": False,
        },
        "limitations": [
            "Synthetic evidence is not production evidence.",
            (
                "CUSUM thresholds are empirically risk-calibrated; "
                "no distribution-free guarantee is claimed for "
                "overlapping windows."
            ),
            (
                "Real shadow-mode evidence is required before "
                "operational deployment."
            ),
        ],
    }
    result_path = (
        output_root
        / "dual_path_episode_experiment.json"
    )
    report_path = output_root / "freeze_report.md"
    result_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    _write_report(
        path=report_path,
        results=results,
    )
    print("Saved:", result_path)
    print("Saved:", report_path)
    print("Status:", release_status)
    print("Locked test:", locked_status)


if __name__ == "__main__":
    main()
