from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from statistics import mean, median
from typing import Any, Callable

from .feature_access import true_root_cause_node_id
from .incident_detector import (
    IncidentDetectorModel,
    choose_incident_threshold,
    fit_incident_detector,
    incident_operating_curve,
)
from .learned_fusion import (
    RootCauseFusionModel,
    fit_root_cause_fusion,
    learned_fusion_scores,
)
from .models import BaselineName
from .root_cause_baselines import (
    rank_node_scores,
    run_baseline,
)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((len(ordered) - 1) * fraction)),
    )
    return ordered[index]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _classification_metrics(
    rows: list[tuple[float, int]],
    threshold: float,
) -> dict[str, float]:
    positives = sum(target for _, target in rows)
    negatives = len(rows) - positives
    true_positives = sum(
        score >= threshold and bool(target)
        for score, target in rows
    )
    false_positives = sum(
        score >= threshold and not target
        for score, target in rows
    )
    return {
        "precision": _safe_divide(
            true_positives,
            true_positives + false_positives,
        ),
        "recall": _safe_divide(true_positives, positives),
        "healthy_false_selection_rate": _safe_divide(
            false_positives,
            negatives,
        ),
    }


def _select_threshold(
    rows: list[tuple[float, int]],
    maximum_false_selection_rate: float = 0.10,
    minimum_coverage: float = 0.70,
) -> dict[str, float | bool]:
    candidates = sorted(
        {
            score
            for score, _ in rows
        }
        | {
            min(score for score, _ in rows) - 1e-9,
            max(score for score, _ in rows) + 1e-9,
        }
    )
    ranked = []
    for threshold in candidates:
        metrics = _classification_metrics(rows, threshold)
        violation = max(
            0.0,
            metrics["healthy_false_selection_rate"]
            - maximum_false_selection_rate,
        ) + max(
            0.0,
            minimum_coverage - metrics["recall"],
        )
        feasible = (
            metrics["healthy_false_selection_rate"]
            <= maximum_false_selection_rate
            and metrics["recall"] >= minimum_coverage
        )
        ranked.append(
            (
                feasible,
                -violation,
                metrics["recall"]
                - metrics["healthy_false_selection_rate"],
                -threshold,
                threshold,
                metrics,
            )
        )
    best = max(ranked)
    return {
        "threshold": best[4],
        "feasible": best[0],
        **best[5],
    }


def _cohort_summary(
    records: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    output = []
    for cohort, rows in grouped.items():
        negatives = [
            row for row in rows if not row["target"]
        ]
        positives = [
            row for row in rows if row["target"]
        ]
        output.append(
            {
                "cohort": cohort,
                "window_count": len(rows),
                "negative_count": len(negatives),
                "positive_count": len(positives),
                "false_positive_count": sum(
                    row["selected"] for row in negatives
                ),
                "healthy_false_selection_rate": _safe_divide(
                    sum(row["selected"] for row in negatives),
                    len(negatives),
                ),
                "recall": _safe_divide(
                    sum(row["selected"] for row in positives),
                    len(positives),
                ),
                "mean_probability": mean(
                    row["score"] for row in rows
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -row["healthy_false_selection_rate"],
            -row["window_count"],
            row["cohort"],
        ),
    )


def _calibration(
    records: list[dict[str, Any]],
    bin_count: int = 10,
) -> list[dict[str, float | int]]:
    output = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        rows = [
            row
            for row in records
            if lower <= row["score"] < upper
            or (
                index == bin_count - 1
                and row["score"] == 1.0
            )
        ]
        if rows:
            output.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": len(rows),
                    "mean_probability": mean(
                        row["score"] for row in rows
                    ),
                    "observed_rate": mean(
                        row["target"] for row in rows
                    ),
                }
            )
    return output


def _cohort_normalized_scores(
    *,
    windows: list[dict[str, Any]],
    model: IncidentDetectorModel,
    scenario_metadata: dict[str, dict[str, Any]],
) -> dict[str, float]:
    by_cutoff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        by_cutoff[str(window["observation_cutoff"])].append(window)
    history: dict[str, list[float]] = defaultdict(list)
    output: dict[str, float] = {}
    for cutoff in sorted(by_cutoff):
        pending: list[tuple[str, float]] = []
        for window in by_cutoff[cutoff]:
            scenario_id = str(window["source_scenario_id"])
            topology = str(
                scenario_metadata[scenario_id][
                    "topology_fingerprint"
                ]
            )
            probability = model.predict_score(window)
            logit = math.log(
                max(1e-9, probability)
                / max(1e-9, 1.0 - probability)
            )
            previous = history[topology]
            if len(previous) >= 4:
                center = median(previous)
                deviations = [
                    abs(value - center)
                    for value in previous
                ]
                scale = max(
                    0.25,
                    1.4826 * median(deviations),
                )
                normalized = (
                    1.0
                    / (
                        1.0
                        + math.exp(
                            -max(
                                -50.0,
                                min(
                                    50.0,
                                    (logit - center)
                                    / scale,
                                ),
                            )
                        )
                    )
                )
            else:
                normalized = probability
            output[str(window["window_id"])] = normalized
            pending.append((topology, logit))
        # Only prior cutoffs are used, preventing same-window pair leakage.
        for topology, logit in pending:
            history[topology].append(logit)
    return output


def stage1_diagnostics(
    *,
    windows_by_split: dict[str, list[dict[str, Any]]],
    scenario_metadata: dict[str, dict[str, Any]],
) -> tuple[IncidentDetectorModel, dict[str, Any]]:
    model = fit_incident_detector(
        training_windows=windows_by_split["train"],
        scenario_metadata=scenario_metadata,
        use_counterfactual_pairs=True,
        cohort_balancing=True,
    )
    validation_curve = incident_operating_curve(
        windows=windows_by_split["validation"],
        model=model,
    )
    for row in validation_curve:
        threshold = float(row["threshold"])
        metrics = _classification_metrics(
            [
                (
                    model.predict_score(window),
                    int(window["targets"]["current_incident"]),
                )
                for window in windows_by_split["validation"]
            ],
            threshold,
        )
        row["precision"] = metrics["precision"]
    selected = choose_incident_threshold(
        validation_windows=windows_by_split["validation"],
        model=model,
        scenario_metadata=scenario_metadata,
        maximum_false_selection_rate=0.10,
        minimum_incident_coverage=0.70,
        maximum_cohort_false_selection_rate=0.15,
    )
    selection = {
        "threshold": selected.threshold,
        "feasible": selected.feasible,
        "precision": _classification_metrics(
            [
                (
                    model.predict_score(window),
                    int(window["targets"]["current_incident"]),
                )
                for window in windows_by_split["validation"]
            ],
            selected.threshold,
        )["precision"],
        "recall": selected.coverage,
        "healthy_false_selection_rate": (
            selected.false_selection_rate
        ),
        "maximum_cohort_false_selection_rate": (
            selected.maximum_cohort_false_selection_rate
        ),
        "cohort_false_selection_rates": dict(
            selected.cohort_false_selection_rates
        ),
    }
    threshold = float(selection["threshold"])

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split, windows in windows_by_split.items():
        records = []
        for window in windows:
            scenario_id = str(window["source_scenario_id"])
            metadata = scenario_metadata[scenario_id]
            score = model.predict_score(window)
            target = int(
                window["targets"]["current_incident"]
            )
            records.append(
                {
                    "window_id": str(window["window_id"]),
                    "scenario_id": scenario_id,
                    "score": score,
                    "target": target,
                    "selected": score >= threshold,
                    "topology": metadata[
                        "topology_fingerprint"
                    ],
                    "confounders": "+".join(
                        sorted(metadata["confounders"])
                    )
                    or "none",
                    "missingness": metadata[
                        "missingness_mode"
                    ],
                    "regime": (
                        "healthy_scenario"
                        if metadata["healthy"]
                        else (
                            "active_fault"
                            if target
                            else "pre_incident_fault_scenario"
                        )
                    ),
                    "failure_type": metadata["failure_type"],
                }
            )
        split_records[split] = records

    coefficients = sorted(
        [
            {
                "feature": name,
                "coefficient": weight,
                "absolute_coefficient": abs(weight),
            }
            for name, weight in zip(
                model.feature_names,
                model.weights,
                strict=True,
            )
        ],
        key=lambda row: -row["absolute_coefficient"],
    )

    contribution_rows = []
    for split in ("validation", "test"):
        for window, record in zip(
            windows_by_split[split],
            split_records[split],
            strict=True,
        ):
            if record["target"] or not record["selected"]:
                continue
            standardized = model.standardized_vector(window)
            contributions = [
                weight * value
                for weight, value in zip(
                    model.weights,
                    standardized,
                    strict=True,
                )
            ]
            contribution_rows.append(
                {
                    "split": split,
                    "window_id": record["window_id"],
                    "scenario_id": record["scenario_id"],
                    "score": record["score"],
                    "top_contributions": [
                        {
                            "feature": model.feature_names[index],
                            "contribution": contributions[index],
                        }
                        for index in sorted(
                            range(len(contributions)),
                            key=lambda index: -abs(
                                contributions[index]
                            ),
                        )[:8]
                    ],
                }
            )

    drift = {}
    for split, windows in windows_by_split.items():
        standardized = [
            model.standardized_vector(window)
            for window in windows
        ]
        drift[split] = sorted(
            [
                {
                    "feature": name,
                    "standardized_mean": mean(
                        row[index] for row in standardized
                    ),
                    "mean_absolute_z": mean(
                        abs(row[index])
                        for row in standardized
                    ),
                }
                for index, name in enumerate(
                    model.feature_names
                )
            ],
            key=lambda row: -abs(row["standardized_mean"]),
        )

    cohort_normalization = {}
    for split in ("validation", "test", "ood_test"):
        scores = _cohort_normalized_scores(
            windows=windows_by_split[split],
            model=model,
            scenario_metadata=scenario_metadata,
        )
        cohort_normalization[split] = [
            (
                scores[str(window["window_id"])],
                int(window["targets"]["current_incident"]),
            )
            for window in windows_by_split[split]
        ]
    normalized_selection = _select_threshold(
        cohort_normalization["validation"]
    )

    return model, {
        "selection": selection,
        "precision_recall_curve": validation_curve,
        "split_metrics": {
            split: _classification_metrics(
                [
                    (record["score"], record["target"])
                    for record in records
                ],
                threshold,
            )
            for split, records in split_records.items()
        },
        "false_positive_cohorts": {
            split: {
                "topology": _cohort_summary(
                    records,
                    lambda row: str(row["topology"]),
                ),
                "confounders": _cohort_summary(
                    records,
                    lambda row: str(row["confounders"]),
                ),
                "missingness": _cohort_summary(
                    records,
                    lambda row: str(row["missingness"]),
                ),
                "fault_free_regime": _cohort_summary(
                    records,
                    lambda row: str(row["regime"]),
                ),
            }
            for split, records in split_records.items()
        },
        "calibration": {
            split: _calibration(records)
            for split, records in split_records.items()
        },
        "calibration_by_topology": {
            split: _cohort_summary(
                records,
                lambda row: str(row["topology"]),
            )
            for split, records in split_records.items()
        },
        "feature_coefficients": coefficients,
        "false_positive_feature_contributions": (
            contribution_rows
        ),
        "feature_drift": drift,
        "cohort_normalization_ablation": {
            "selection": normalized_selection,
            "split_metrics": {
                split: _classification_metrics(
                    rows,
                    float(normalized_selection["threshold"]),
                )
                for split, rows in cohort_normalization.items()
            },
            "causal_baseline": (
                "Only scores from prior observation cutoffs in the "
                "same topology family are used."
            ),
        },
        "model_design": {
            "observable_impact_labeling": True,
            "causal_historical_features": True,
            "counterfactual_pairwise_training": True,
            "confounder_topology_balancing": True,
            "global_false_selection_constraint": 0.10,
            "cohort_false_selection_constraint": 0.15,
            "minimum_recall_constraint": 0.70,
        },
    }


def stage1_ablation_suite(
    *,
    windows_by_split: dict[str, list[dict[str, Any]]],
    scenario_metadata: dict[str, dict[str, Any]],
    full_model: IncidentDetectorModel,
) -> list[dict[str, Any]]:
    configurations = [
        (
            "raw",
            {
                "feature_mode": "raw",
                "use_counterfactual_pairs": False,
                "cohort_balancing": False,
            },
        ),
        (
            "historical_features",
            {
                "feature_mode": "full",
                "use_counterfactual_pairs": False,
                "cohort_balancing": False,
            },
        ),
        (
            "historical_plus_counterfactual",
            {
                "feature_mode": "full",
                "use_counterfactual_pairs": True,
                "cohort_balancing": False,
            },
        ),
        (
            "historical_plus_balancing",
            {
                "feature_mode": "full",
                "use_counterfactual_pairs": False,
                "cohort_balancing": True,
            },
        ),
        ("full_stage1_v2", None),
    ]
    results = []
    for name, configuration in configurations:
        model = (
            full_model
            if configuration is None
            else fit_incident_detector(
                training_windows=windows_by_split["train"],
                scenario_metadata=scenario_metadata,
                iterations=400,
                **configuration,
            )
        )
        selection = choose_incident_threshold(
            validation_windows=windows_by_split["validation"],
            model=model,
            scenario_metadata=scenario_metadata,
            maximum_false_selection_rate=0.10,
            minimum_incident_coverage=0.70,
            maximum_cohort_false_selection_rate=0.15,
        )
        split_metrics = {}
        for split in ("validation", "test", "ood_test"):
            rows = [
                (
                    model.predict_score(window),
                    int(window["targets"]["current_incident"]),
                )
                for window in windows_by_split[split]
            ]
            split_metrics[split] = _classification_metrics(
                rows,
                selection.threshold,
            )
        results.append(
            {
                "name": name,
                "feature_count": len(model.feature_names),
                "selection_feasible": selection.feasible,
                "maximum_validation_cohort_false_selection_rate": (
                    selection.maximum_cohort_false_selection_rate
                ),
                "split_metrics": split_metrics,
            }
        )
    return results


def _root_rank(
    *,
    window: dict[str, Any],
    baseline: BaselineName,
    fusion_model: RootCauseFusionModel | None,
) -> int | None:
    root = true_root_cause_node_id(window)
    if root is None:
        return None
    scores = run_baseline(
        window=window,
        baseline=baseline,
        fusion_model=fusion_model,
    )
    ranked = rank_node_scores(scores)
    return [
        item.node_id for item in ranked
    ].index(root) + 1


def _ranking_summary(ranks: list[int]) -> dict[str, float | int]:
    return {
        "window_count": len(ranks),
        "mrr": (
            mean(1.0 / rank for rank in ranks)
            if ranks
            else 0.0
        ),
        "hits_at_1": _safe_divide(
            sum(rank <= 1 for rank in ranks),
            len(ranks),
        ),
        "hits_at_3": _safe_divide(
            sum(rank <= 3 for rank in ranks),
            len(ranks),
        ),
        "mean_rank": mean(ranks) if ranks else 0.0,
    }


def independent_ranking_evaluation(
    *,
    windows_by_split: dict[str, list[dict[str, Any]]],
    incident_model: IncidentDetectorModel,
    incident_threshold: float,
    fusion_model: RootCauseFusionModel,
) -> dict[str, Any]:
    output = {}
    for baseline in BaselineName:
        active_model = (
            fusion_model
            if baseline == BaselineName.LEARNED_FUSION
            else None
        )
        split_output = {}
        for split in ("validation", "test", "ood_test"):
            faulty = [
                window
                for window in windows_by_split[split]
                if int(
                    window["targets"]["current_incident"]
                )
                == 1
            ]
            accepted = [
                window
                for window in faulty
                if (
                    incident_model.predict_score(window)
                    >= incident_threshold
                    or (
                        split == "ood_test"
                        and incident_model.is_out_of_distribution(
                            window
                        )
                    )
                )
            ]
            all_ranks = [
                rank
                for rank in (
                    _root_rank(
                        window=window,
                        baseline=baseline,
                        fusion_model=active_model,
                    )
                    for window in faulty
                )
                if rank is not None
            ]
            accepted_ranks = [
                rank
                for rank in (
                    _root_rank(
                        window=window,
                        baseline=baseline,
                        fusion_model=active_model,
                    )
                    for window in accepted
                )
                if rank is not None
            ]
            split_output[split] = {
                "all_true_faulty": _ranking_summary(all_ranks),
                "stage1_accepted_true_faulty": (
                    _ranking_summary(accepted_ranks)
                ),
                "stage1_recall": _safe_divide(
                    len(accepted),
                    len(faulty),
                ),
                "end_to_end_hits_at_1": _safe_divide(
                    sum(rank <= 1 for rank in accepted_ranks),
                    len(faulty),
                ),
                "end_to_end_hits_at_3": _safe_divide(
                    sum(rank <= 3 for rank in accepted_ranks),
                    len(faulty),
                ),
            }
        output[baseline.value] = split_output
    return output


def _topology_depth(window: dict[str, Any]) -> int:
    root = true_root_cause_node_id(window)
    if root is None:
        return 0
    node_ids = [str(node) for node in window["node_ids"]]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source_index, target_index in window["edge_index"]:
        source = node_ids[source_index]
        target = node_ids[target_index]
        adjacency[source].add(target)
        adjacency[target].add(source)
    distances = {root: 0}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return max(distances.values(), default=0)


def _bootstrap_clustered_mean(
    *,
    values_by_cluster: dict[str, list[float]],
    seed: int,
    repetitions: int = 1_000,
) -> dict[str, float]:
    clusters = sorted(values_by_cluster)
    observed = mean(
        value
        for values in values_by_cluster.values()
        for value in values
    )
    if not clusters:
        return {
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "probability_positive": 0.0,
        }
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        selected = [
            rng.choice(clusters)
            for _ in clusters
        ]
        values = [
            value
            for cluster in selected
            for value in values_by_cluster[cluster]
        ]
        samples.append(mean(values))
    return {
        "mean": observed,
        "ci_lower": _quantile(samples, 0.025),
        "ci_upper": _quantile(samples, 0.975),
        "probability_positive": _safe_divide(
            sum(value > 0.0 for value in samples),
            len(samples),
        ),
    }


def topology_validation(
    *,
    training_windows: list[dict[str, Any]],
    windows_by_split: dict[str, list[dict[str, Any]]],
    scenario_metadata: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71),
) -> dict[str, Any]:
    paired_rows = []
    seed_summaries = []
    for seed in seeds:
        full = fit_root_cause_fusion(
            training_windows=training_windows,
            use_topology=True,
            iterations=120,
            random_seed=seed,
        )
        ablated = fit_root_cause_fusion(
            training_windows=training_windows,
            use_topology=False,
            iterations=120,
            random_seed=seed,
        )
        for split in ("validation", "test", "ood_test"):
            differences = []
            for window in windows_by_split[split]:
                if not int(
                    window["targets"]["current_incident"]
                ):
                    continue
                full_rank = _root_rank(
                    window=window,
                    baseline=BaselineName.LEARNED_FUSION,
                    fusion_model=full,
                )
                ablated_rank = _root_rank(
                    window=window,
                    baseline=BaselineName.LEARNED_FUSION,
                    fusion_model=ablated,
                )
                if full_rank is None or ablated_rank is None:
                    continue
                difference = (
                    1.0 / full_rank
                    - 1.0 / ablated_rank
                )
                scenario_id = str(
                    window["source_scenario_id"]
                )
                metadata = scenario_metadata[scenario_id]
                depth = _topology_depth(window)
                row = {
                    "seed": seed,
                    "split": split,
                    "window_id": str(window["window_id"]),
                    "scenario_id": scenario_id,
                    "topology": metadata[
                        "topology_fingerprint"
                    ],
                    "failure_type": metadata["failure_type"],
                    "topology_depth": depth,
                    "depth_band": (
                        "shallow"
                        if depth <= 2
                        else "medium"
                        if depth <= 4
                        else "deep"
                    ),
                    "rr_difference": difference,
                }
                paired_rows.append(row)
                differences.append(difference)
            seed_summaries.append(
                {
                    "seed": seed,
                    "split": split,
                    "mean_rr_difference": (
                        mean(differences)
                        if differences
                        else 0.0
                    ),
                    "positive_window_fraction": _safe_divide(
                        sum(value > 0 for value in differences),
                        len(differences),
                    ),
                }
            )

    def breakdown(
        field: str,
    ) -> dict[str, dict[str, float]]:
        values: dict[
            str,
            dict[str, list[float]],
        ] = defaultdict(lambda: defaultdict(list))
        for row in paired_rows:
            key = f"{row['split']}::{row[field]}"
            values[key][str(row["topology"])].append(
                float(row["rr_difference"])
            )
        return {
            key: _bootstrap_clustered_mean(
                values_by_cluster=clusters,
                seed=9_001,
            )
            for key, clusters in values.items()
        }

    by_split_clusters: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in paired_rows:
        by_split_clusters[str(row["split"])][
            str(row["topology"])
        ].append(float(row["rr_difference"]))
    split_bootstrap = {
        split: _bootstrap_clustered_mean(
            values_by_cluster=clusters,
            seed=8_003,
        )
        for split, clusters in by_split_clusters.items()
    }
    retain = (
        split_bootstrap.get("test", {}).get(
            "ci_lower",
            -1.0,
        )
        >= 0.0
        or split_bootstrap.get("ood_test", {}).get(
            "ci_lower",
            -1.0,
        )
        >= 0.0
    )
    return {
        "seeds": list(seeds),
        "seed_summaries": seed_summaries,
        "clustered_bootstrap_by_split": split_bootstrap,
        "per_failure_family": breakdown("failure_type"),
        "topology_depth_breakdown": breakdown("depth_band"),
        "retain_topology": retain,
        "retention_rule": (
            "Retain only when the clustered 95% interval is "
            "non-negative on test or OOD test."
        ),
    }


def _distribution_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "minimum": min(values, default=0.0),
        "q05": _quantile(values, 0.05),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "maximum": max(values, default=0.0),
    }


def _auroc(negative: list[float], positive: list[float]) -> float:
    comparisons = [
        1.0
        if positive_value > negative_value
        else 0.5
        if positive_value == negative_value
        else 0.0
        for positive_value in positive
        for negative_value in negative
    ]
    return mean(comparisons) if comparisons else 0.0


def _average_precision(
    negative: list[float],
    positive: list[float],
) -> float:
    rows = sorted(
        [
            (value, 0) for value in negative
        ]
        + [
            (value, 1) for value in positive
        ],
        reverse=True,
    )
    true_positives = 0
    precisions = []
    for index, (_, label) in enumerate(rows, start=1):
        if label:
            true_positives += 1
            precisions.append(true_positives / index)
    return mean(precisions) if precisions else 0.0


def _ks_statistic(left: list[float], right: list[float]) -> float:
    points = sorted(set(left + right))
    return max(
        (
            abs(
                _safe_divide(
                    sum(value <= point for value in left),
                    len(left),
                )
                - _safe_divide(
                    sum(value <= point for value in right),
                    len(right),
                )
            )
            for point in points
        ),
        default=0.0,
    )


def ood_audit(
    *,
    windows_by_split: dict[str, list[dict[str, Any]]],
    model: IncidentDetectorModel,
    scenario_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    distances = {
        split: [
            model.ood_distance(window)
            for window in windows
        ]
        for split, windows in windows_by_split.items()
    }
    in_distribution = (
        distances["train"]
        + distances["validation"]
        + distances["test"]
    )
    ood_values = distances["ood_test"]
    fixed_rates = {}
    for false_positive_rate in (0.01, 0.05, 0.10):
        threshold = _quantile(
            in_distribution,
            1.0 - false_positive_rate,
        )
        fixed_rates[str(false_positive_rate)] = {
            "threshold": threshold,
            "observed_id_false_positive_rate": _safe_divide(
                sum(
                    value > threshold
                    for value in in_distribution
                ),
                len(in_distribution),
            ),
            "ood_detection_rate": _safe_divide(
                sum(value > threshold for value in ood_values),
                len(ood_values),
            ),
        }

    cohort_distances: dict[str, list[float]] = defaultdict(list)
    for window, distance in zip(
        windows_by_split["ood_test"],
        ood_values,
        strict=True,
    ):
        metadata = scenario_metadata[
            str(window["source_scenario_id"])
        ]
        key = (
            f"missingness={metadata['missingness_mode']}"
            f"|rooms={metadata['room_count']}"
            f"|floors={metadata['floor_count']}"
        )
        cohort_distances[key].append(distance)

    return {
        "distance_distributions": {
            split: _distribution_summary(values)
            for split, values in distances.items()
        },
        "ood_auroc": _auroc(
            in_distribution,
            ood_values,
        ),
        "ood_average_precision": _average_precision(
            in_distribution,
            ood_values,
        ),
        "ks_ood_vs_train": _ks_statistic(
            distances["train"],
            ood_values,
        ),
        "fixed_id_false_positive_rates": fixed_rates,
        "configured_threshold": model.ood_distance_threshold,
        "configured_detection_rates": {
            split: _safe_divide(
                sum(
                    value > model.ood_distance_threshold
                    for value in values
                ),
                len(values),
            )
            for split, values in distances.items()
        },
        "ood_dimension_breakdown": {
            cohort: _distribution_summary(values)
            for cohort, values in cohort_distances.items()
        },
        "aligned_with_ood_definition": (
            _auroc(in_distribution, ood_values) >= 0.70
        ),
    }
