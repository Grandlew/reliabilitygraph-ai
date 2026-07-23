import pytest

from app.domain.nrim.baselines.abstention import (
    calculate_score_margin,
    decide_abstention,
    threshold_candidates,
)
from app.domain.nrim.baselines.models import (
    NodeScore,
)


def scores():
    return [
        NodeScore(
            node_id="a",
            score=0.8,
        ),
        NodeScore(
            node_id="b",
            score=0.5,
        ),
    ]


def test_margin() -> None:
    assert calculate_score_margin(
        scores()
    ) == pytest.approx(0.3)


def test_abstains_on_low_top_score() -> None:
    decision = decide_abstention(
        ranked_scores=scores(),
        minimum_top_score=0.9,
    )

    assert decision.abstain is True


def test_abstains_on_small_margin() -> None:
    decision = decide_abstention(
        ranked_scores=scores(),
        minimum_top_score=0.5,
        minimum_margin=0.4,
    )

    assert decision.abstain is True


def test_accepts_strong_ranking() -> None:
    decision = decide_abstention(
        ranked_scores=scores(),
        minimum_top_score=0.5,
        minimum_margin=0.2,
    )

    assert decision.abstain is False


def test_threshold_candidates_cover_extremes() -> None:
    candidates = threshold_candidates(
        [0.2, 0.5, 0.8]
    )

    assert min(candidates) < 0.2
    assert max(candidates) > 0.8
