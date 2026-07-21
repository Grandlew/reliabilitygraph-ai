from __future__ import annotations

from dataclasses import dataclass

from .models import NodeScore


@dataclass(frozen=True)
class AbstentionDecision:
    abstain: bool
    top_score: float
    score_margin: float
    reason: str


def calculate_score_margin(
    ranked_scores: list[NodeScore],
) -> float:
    if not ranked_scores:
        raise ValueError(
            "Cannot calculate margin for empty scores."
        )

    if len(ranked_scores) == 1:
        return max(
            0.0,
            ranked_scores[0].score,
        )

    return max(
        0.0,
        ranked_scores[0].score
        - ranked_scores[1].score,
    )


def decide_abstention(
    *,
    ranked_scores: list[NodeScore],
    minimum_top_score: float,
    minimum_margin: float = 0.0,
) -> AbstentionDecision:
    if not ranked_scores:
        raise ValueError(
            "Cannot abstain on empty node ranking."
        )

    top_score = float(
        ranked_scores[0].score
    )
    margin = calculate_score_margin(
        ranked_scores
    )

    if top_score < minimum_top_score:
        return AbstentionDecision(
            abstain=True,
            top_score=top_score,
            score_margin=margin,
            reason=f"Top score {top_score:.4f} is below minimum {minimum_top_score:.4f}"
        )

    if margin < minimum_margin:
        return AbstentionDecision(
            abstain=True,
            top_score=top_score,
            score_margin=margin,
            reason=f"Score margin {margin:.4f} is below minimum {minimum_margin:.4f}"
        )

    return AbstentionDecision(
        abstain=False,
        top_score=top_score,
        score_margin=margin,
        reason="Thresholds met"
    )


def threshold_candidates(
    top_scores: list[float],
) -> list[float]:
    unique_scores = sorted(set(top_scores))

    if not unique_scores:
        return [0.0]

    candidates = [
        unique_scores[0] - 1e-9,
        unique_scores[-1] + 1e-9,
    ]

    candidates.extend(
        (
            unique_scores[index]
            + unique_scores[index + 1]
        )
        / 2.0
        for index in range(
            len(unique_scores) - 1
        )
    )

    return sorted(set(candidates))
