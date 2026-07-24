from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from .feature_access import true_root_cause_node_id
from .models import NodeScore
from .root_cause_baselines import (
    anomaly_baseline,
    rank_node_scores,
)
from .topology_scoring import topology_propagation_baseline


@dataclass(frozen=True)
class PropagationDiagnostics:
    window_count: int
    top1_change_rate: float
    mean_pairwise_inversions: float
    mean_root_rank_change: float
    mean_topology_increment: float
    score_saturation_rate: float


def _rank_map(scores: list[NodeScore]) -> dict[str, int]:
    return {
        item.node_id: index
        for index, item in enumerate(
            rank_node_scores(scores),
            start=1,
        )
    }


def summarize_propagation_activity(
    windows: list[dict[str, Any]],
) -> PropagationDiagnostics:
    top1_changes = 0
    inversions: list[float] = []
    root_rank_changes: list[float] = []
    increments: list[float] = []
    saturated = 0
    score_count = 0

    for window in windows:
        local = anomaly_baseline(window=window)
        propagated = topology_propagation_baseline(
            window=window,
            local_scores=local,
        )
        local_ranks = _rank_map(local)
        propagated_ranks = _rank_map(propagated)
        if min(local_ranks, key=local_ranks.get) != min(
            propagated_ranks,
            key=propagated_ranks.get,
        ):
            top1_changes += 1

        node_ids = list(local_ranks)
        pair_count = 0
        inversion_count = 0
        for left_index, left in enumerate(node_ids):
            for right in node_ids[left_index + 1 :]:
                pair_count += 1
                if (
                    local_ranks[left] - local_ranks[right]
                ) * (
                    propagated_ranks[left]
                    - propagated_ranks[right]
                ) < 0:
                    inversion_count += 1
        inversions.append(
            inversion_count / pair_count
            if pair_count
            else 0.0
        )

        root = true_root_cause_node_id(window)
        if root is not None:
            root_rank_changes.append(
                float(local_ranks[root] - propagated_ranks[root])
            )

        for item in propagated:
            increments.append(
                float(item.evidence.get("topology_increment", 0.0))
            )
            saturated += int(item.score >= 1.0 - 1e-12)
            score_count += 1

    count = len(windows)
    return PropagationDiagnostics(
        window_count=count,
        top1_change_rate=top1_changes / count if count else 0.0,
        mean_pairwise_inversions=mean(inversions) if inversions else 0.0,
        mean_root_rank_change=(
            mean(root_rank_changes)
            if root_rank_changes
            else 0.0
        ),
        mean_topology_increment=mean(increments) if increments else 0.0,
        score_saturation_rate=(
            saturated / score_count
            if score_count
            else 0.0
        ),
    )
