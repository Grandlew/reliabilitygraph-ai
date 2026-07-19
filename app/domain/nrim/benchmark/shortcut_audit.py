from __future__ import annotations

from collections import Counter
from typing import Callable

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
