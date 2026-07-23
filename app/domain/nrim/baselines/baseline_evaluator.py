from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .abstention import (
    decide_abstention,
    threshold_candidates,
)
from .feature_access import (
    true_root_cause_node_id,
)
from .models import (
    BaselineName,
    WindowRankingResult,
)
from .ranking_metrics import (
    summarize_ranking_results,
)
from .root_cause_baselines import (
    rank_node_scores,
    run_baseline,
)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_window(
    *,
    window: dict[str, Any],
    baseline: BaselineName,
    minimum_top_score: float,
    minimum_margin: float = 0.0,
    random_seed: int = 42,
) -> WindowRankingResult:
    start = time.perf_counter()

    scores = run_baseline(
        window=window,
        baseline=baseline,
        random_seed=random_seed,
    )

    ranked_scores = rank_node_scores(
        scores
    )

    decision = decide_abstention(
        ranked_scores=ranked_scores,
        minimum_top_score=minimum_top_score,
        minimum_margin=minimum_margin,
    )

    true_node_id = true_root_cause_node_id(
        window
    )

    ranked_node_ids = [
        item.node_id
        for item in ranked_scores
    ]

    true_rank = None

    if (
        true_node_id is not None
        and not decision.abstain
    ):
        true_rank = (
            ranked_node_ids.index(
                true_node_id
            )
            + 1
        )

    runtime_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return WindowRankingResult(
        window_id=str(window["window_id"]),
        split=str(window["split"]),
        baseline=baseline,
        node_scores=ranked_scores,
        ranked_node_ids=ranked_node_ids,
        true_root_cause_node_id=true_node_id,
        true_root_cause_rank=true_rank,
        abstained=decision.abstain,
        top_score=decision.top_score,
        score_margin=decision.score_margin,
        runtime_ms=runtime_ms,
    )


def load_split_windows(
    *,
    manifest: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    return [
        load_json(record["path"])
        for record in manifest["records"]
        if str(record["split"]) == split
    ]


def choose_abstention_threshold(
    *,
    validation_windows: list[dict[str, Any]],
    baseline: BaselineName,
    maximum_healthy_false_selection_rate: float = 0.10,
    minimum_faulty_coverage: float = 0.70,
    random_seed: int = 42,
) -> float:
    # Evaluate base rankings for validation windows
    base_results = []
    for window in validation_windows:
        scores = run_baseline(
            window=window, baseline=baseline, random_seed=random_seed)
        ranked = rank_node_scores(scores)
        base_results.append((window, ranked))

    # Generate candidate thresholds
    top_scores = [r[0].score for _, r in base_results if r]
    candidates = threshold_candidates(top_scores)

    feasible_candidates = []

    for threshold in candidates:
        results = []
        for window, ranked in base_results:
            decision = decide_abstention(
                ranked_scores=ranked, minimum_top_score=threshold)
            true_id = true_root_cause_node_id(window)
            ranked_ids = [item.node_id for item in ranked]

            true_rank = (ranked_ids.index(true_id) +
                         1) if true_id and not decision.abstain else None

            results.append(WindowRankingResult(
                window_id=str(window["window_id"]), split="validation", baseline=baseline,
                node_scores=ranked, ranked_node_ids=ranked_ids, true_root_cause_node_id=true_id,
                true_root_cause_rank=true_rank, abstained=decision.abstain, top_score=decision.top_score,
                score_margin=decision.score_margin, runtime_ms=0.0
            ))

        metrics = summarize_ranking_results(results)

        if (metrics.healthy_false_selection_rate <= maximum_healthy_false_selection_rate and
                metrics.faulty_coverage >= minimum_faulty_coverage):
            # Tie breaking: (MRR, faulty_coverage, -threshold)
            feasible_candidates.append(
                (metrics.mrr, metrics.faulty_coverage, -threshold, threshold))

    if not feasible_candidates:
        raise ValueError(
            "No feasible threshold found satisfying the specified constraints.")

    feasible_candidates.sort(reverse=True)
    return feasible_candidates[0][3]


def evaluate_split(
    *,
    windows: list[dict[str, Any]],
    baseline: BaselineName,
    abstention_threshold: float,
    random_seed: int = 42,
):
    results = [
        evaluate_window(
            window=window,
            baseline=baseline,
            minimum_top_score=(
                abstention_threshold
            ),
            random_seed=random_seed,
        )
        for window in windows
    ]

    return (
        results,
        summarize_ranking_results(results),
    )
