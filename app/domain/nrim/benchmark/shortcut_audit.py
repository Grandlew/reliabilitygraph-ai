from __future__ import annotations

from collections import Counter
import hashlib
from statistics import mean
from typing import Callable

from app.domain.nrim.baselines.root_cause_baselines import (
    NODE_TYPE_CRITICALITY,
)

from .models import ShortcutResult, WindowSummary


ScalarExtractor = Callable[
    [WindowSummary],
    float,
]


SHORTCUTS: dict[str, ScalarExtractor] = {
    "node_count": lambda item: float(
        item.node_count
    ),
    "edge_count": lambda item: float(
        item.edge_count
    ),
    "node_feature_count": lambda item: float(
        item.node_feature_count
    ),
    "edge_feature_count": lambda item: float(
        item.edge_feature_count
    ),
    "missing_feature_fraction": lambda item: float(
        item.missing_feature_fraction
    ),
}


def balanced_accuracy(
    targets: list[int],
    predictions: list[int],
) -> float:
    recalls = []

    for target_class in sorted(set(targets)):
        indices = [
            index
            for index, value in enumerate(targets)
            if value == target_class
        ]

        if not indices:
            continue

        correct = sum(
            predictions[index] == target_class
            for index in indices
        )

        recalls.append(
            correct / len(indices)
        )

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)


def majority_baseline_score(
    targets: list[int],
) -> float:
    if not targets:
        return 0.0

    majority_class = Counter(
        targets
    ).most_common(1)[0][0]

    predictions = [
        majority_class
        for _ in targets
    ]

    return balanced_accuracy(
        targets,
        predictions,
    )


def candidate_thresholds(
    values: list[float],
) -> list[float]:
    unique_values = sorted(set(values))

    if len(unique_values) < 2:
        return []

    return [
        (
            unique_values[index]
            + unique_values[index + 1]
        )
        / 2.0
        for index in range(
            len(unique_values) - 1
        )
    ]


def fit_best_threshold(
    *,
    values: list[float],
    targets: list[int],
) -> tuple[
    float,
    str,
    float,
]:
    thresholds = candidate_thresholds(values)

    if not thresholds:
        return (
            values[0] if values else 0.0,
            "greater_equal",
            majority_baseline_score(targets),
        )

    best_threshold = thresholds[0]
    best_direction = "greater_equal"
    best_score = -1.0

    for threshold in thresholds:
        for direction in (
            "greater_equal",
            "less_equal",
        ):
            if direction == "greater_equal":
                predictions = [
                    int(value >= threshold)
                    for value in values
                ]
            else:
                predictions = [
                    int(value <= threshold)
                    for value in values
                ]

            score = balanced_accuracy(
                targets,
                predictions,
            )

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_direction = direction

    return (
        best_threshold,
        best_direction,
        best_score,
    )


def apply_threshold(
    *,
    values: list[float],
    threshold: float,
    direction: str,
) -> list[int]:
    if direction == "greater_equal":
        return [
            int(value >= threshold)
            for value in values
        ]

    if direction == "less_equal":
        return [
            int(value <= threshold)
            for value in values
        ]

    raise ValueError(
        f"Unsupported threshold direction: {direction}"
    )


def run_shortcut_audit(
    summaries: list[WindowSummary],
    *,
    target_name: str = "future_incident",
    suspicious_balanced_accuracy: float = 0.70,
    suspicious_improvement: float = 0.15,
) -> list[ShortcutResult]:
    training = [
        item
        for item in summaries
        if item.split == "train"
    ]

    validation = [
        item
        for item in summaries
        if item.split == "validation"
    ]

    if not training or not validation:
        raise ValueError(
            "Shortcut audit requires train "
            "and validation windows."
        )

    train_targets = [
        int(getattr(item, target_name))
        for item in training
    ]

    validation_targets = [
        int(getattr(item, target_name))
        for item in validation
    ]

    baseline = majority_baseline_score(
        validation_targets
    )

    results: list[ShortcutResult] = []

    for shortcut_name, extractor in (
        SHORTCUTS.items()
    ):
        train_values = [
            extractor(item)
            for item in training
        ]

        threshold, direction, _ = (
            fit_best_threshold(
                values=train_values,
                targets=train_targets,
            )
        )

        validation_values = [
            extractor(item)
            for item in validation
        ]

        predictions = apply_threshold(
            values=validation_values,
            threshold=threshold,
            direction=direction,
        )

        score = balanced_accuracy(
            validation_targets,
            predictions,
        )

        improvement = score - baseline

        suspicious = (
            score >= suspicious_balanced_accuracy
            and improvement >= suspicious_improvement
        )

        results.append(
            ShortcutResult(
                shortcut_name=shortcut_name,
                target_name=target_name,
                baseline_score=round(
                    baseline,
                    6,
                ),
                shortcut_score=round(
                    score,
                    6,
                ),
                score_improvement=round(
                    improvement,
                    6,
                ),
                threshold=threshold,
                direction=direction,
                suspicious=suspicious,
                explanation=(
                    "Suspicious nuisance-feature "
                    "predictability detected."
                    if suspicious
                    else (
                        "No strong validation shortcut "
                        "detected under this scalar audit."
                    )
                ),
            )
        )

    return results


def _root_index(window: dict) -> int | None:
    labels = [
        int(value)
        for value in window["targets"]["root_cause_node"]
    ]
    positives = [
        index
        for index, value in enumerate(labels)
        if value == 1
    ]
    return positives[0] if len(positives) == 1 else None


def _node_types(window: dict) -> list[str]:
    names = list(window["node_feature_names"])
    type_indices = [
        (index, name.removeprefix("node_type__"))
        for index, name in enumerate(names)
        if name.startswith("node_type__")
    ]
    return [
        next(
            (
                node_type
                for index, node_type in type_indices
                if float(row[index]) > 0.5
            ),
            "unknown",
        )
        for row in window["node_features"]
    ]


def _feature_values(
    window: dict,
    name: str,
) -> list[float]:
    names = list(window["node_feature_names"])
    if name not in names:
        return [0.0 for _ in window["node_ids"]]
    index = names.index(name)
    return [
        float(row[index])
        for row in window["node_features"]
    ]


def _identifier_bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 8


def run_root_cause_shortcut_audit(
    *,
    training_windows: list[dict],
    validation_windows: list[dict],
    suspicious_hits_at_1: float = 0.70,
    suspicious_improvement: float = 0.15,
) -> list[ShortcutResult]:
    """Audit fixed and learned nuisance-only root-cause rankings."""
    training_faulty = [
        window
        for window in training_windows
        if _root_index(window) is not None
    ]
    validation_faulty = [
        window
        for window in validation_windows
        if _root_index(window) is not None
    ]
    if not training_faulty or not validation_faulty:
        raise ValueError(
            "Root-cause shortcut audit requires faulty train "
            "and validation windows."
        )

    root_types = Counter(
        _node_types(window)[int(_root_index(window))]
        for window in training_faulty
    )
    default_root_type = root_types.most_common(1)[0][0]
    root_positions = Counter(
        int(_root_index(window))
        for window in training_faulty
    )
    default_position = root_positions.most_common(1)[0][0]
    root_ids = Counter(
        str(window["node_ids"][int(_root_index(window))])
        for window in training_faulty
    )

    type_by_size: dict[int, Counter[str]] = {}
    type_by_bucket: dict[int, Counter[str]] = {}
    for window in training_faulty:
        root_type = _node_types(window)[int(_root_index(window))]
        type_by_size.setdefault(
            len(window["node_ids"]),
            Counter(),
        )[root_type] += 1
        bucket = _identifier_bucket(
            str(window["source_scenario_id"])
        )
        type_by_bucket.setdefault(bucket, Counter())[root_type] += 1

    baseline = mean(
        1.0 / len(window["node_ids"])
        for window in validation_faulty
    )

    def predicted_type(window: dict, source: str) -> str:
        if source == "topology_size":
            counts = type_by_size.get(len(window["node_ids"]))
        else:
            counts = type_by_bucket.get(
                _identifier_bucket(
                    str(window["source_scenario_id"])
                )
            )
        return (
            counts.most_common(1)[0][0]
            if counts
            else default_root_type
        )

    def top_index(window: dict, shortcut: str) -> int:
        node_ids = [str(value) for value in window["node_ids"]]
        node_types = _node_types(window)

        if shortcut == "node_type":
            scores = [
                float(node_type == default_root_type)
                for node_type in node_types
            ]
        elif shortcut == "static_criticality":
            scores = [
                NODE_TYPE_CRITICALITY.get(node_type, 0.0)
                for node_type in node_types
            ]
        elif shortcut == "node_degree":
            scores = _feature_values(window, "total_degree")
        elif shortcut == "node_position":
            scores = [
                -abs(index - default_position)
                for index in range(len(node_ids))
            ]
        elif shortcut == "node_identifier":
            scores = [
                float(root_ids[node_id])
                for node_id in node_ids
            ]
        elif shortcut in {"topology_size", "scenario_identifier"}:
            expected_type = predicted_type(window, shortcut)
            scores = [
                float(node_type == expected_type)
                for node_type in node_types
            ]
        else:
            raise ValueError(f"Unknown root-cause shortcut: {shortcut}")

        return min(
            range(len(node_ids)),
            key=lambda index: (-scores[index], node_ids[index]),
        )

    results = []
    for shortcut in (
        "node_type",
        "static_criticality",
        "node_degree",
        "node_position",
        "node_identifier",
        "topology_size",
        "scenario_identifier",
    ):
        score = mean(
            float(top_index(window, shortcut) == _root_index(window))
            for window in validation_faulty
        )
        improvement = score - baseline
        suspicious = (
            score >= suspicious_hits_at_1
            and improvement >= suspicious_improvement
        )
        results.append(
            ShortcutResult(
                shortcut_name=f"root_cause__{shortcut}",
                target_name="root_cause_node",
                baseline_score=round(baseline, 6),
                shortcut_score=round(score, 6),
                score_improvement=round(improvement, 6),
                suspicious=suspicious,
                explanation=(
                    "Suspicious root-cause shortcut detected."
                    if suspicious
                    else "No blocking root-cause shortcut detected."
                ),
            )
        )

    return results
