from __future__ import annotations

from statistics import mean, median

from .models import (
    RankingMetricSummary,
    WindowRankingResult,
)


def reciprocal_rank(
    rank: int | None,
) -> float:
    if rank is None:
        return 0.0

    if rank < 1:
        raise ValueError(
            "Rank must be at least 1."
        )

    return 1.0 / rank


def hits_at_k(
    rank: int | None,
    *,
    k: int,
) -> float:
    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    return float(
        rank is not None and rank <= k
    )


def summarize_ranking_results(
    results: list[WindowRankingResult],
) -> RankingMetricSummary:
    faulty_results = [
        r for r in results if r.true_root_cause_node_id is not None]
    healthy_results = [r for r in results if r.true_root_cause_node_id is None]

    # Faulty window metrics
    mrr_values = []
    hits1_values = []
    hits3_values = []
    ranks = []
    selective_mrr_values = []
    faulty_non_abstained = 0

    for r in faulty_results:
        # Contribution to standard metrics (abstained = 0 score)
        rank = r.true_root_cause_rank if not r.abstained else None
        mrr_values.append(reciprocal_rank(rank))
        hits1_values.append(hits_at_k(rank, k=1))
        hits3_values.append(hits_at_k(rank, k=3))

        if not r.abstained:
            faulty_non_abstained += 1
            if r.true_root_cause_rank is not None:
                ranks.append(float(r.true_root_cause_rank))
                selective_mrr_values.append(
                    reciprocal_rank(r.true_root_cause_rank))

    # Healthy window metrics
    healthy_abstained_count = sum(1 for r in healthy_results if r.abstained)

    return RankingMetricSummary(
        window_count=len(results),
        evaluated_faulty_windows=len(faulty_results),
        mrr=mean(mrr_values) if mrr_values else 0.0,
        hits_at_1=mean(hits1_values) if hits1_values else 0.0,
        hits_at_3=mean(hits3_values) if hits3_values else 0.0,
        mean_rank=mean(ranks) if ranks else None,
        median_rank=median(ranks) if ranks else None,
        healthy_window_count=len(healthy_results),
        healthy_abstention_rate=(
            healthy_abstained_count / len(healthy_results)
            if healthy_results else 0.0
        ),
        healthy_false_selection_rate=(
            (len(healthy_results) - healthy_abstained_count) / len(healthy_results)
            if healthy_results else 0.0
        ),
        faulty_coverage=(
            faulty_non_abstained / len(faulty_results)
            if faulty_results else 0.0
        ),
        selective_mrr=(
            mean(selective_mrr_values) if selective_mrr_values else 0.0
        ),
        mean_runtime_ms=(
            mean([r.runtime_ms for r in results])
            if results else 0.0
        )
    )
