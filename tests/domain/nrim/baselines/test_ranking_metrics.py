from app.domain.nrim.baselines.models import (
    BaselineName,
    NodeScore,
    WindowRankingResult,
)
from app.domain.nrim.baselines.ranking_metrics import (
    hits_at_k,
    reciprocal_rank,
    summarize_ranking_results,
)


def make_result(
    *,
    window_id: str,
    true_node: str | None,
    rank: int | None,
    abstained: bool,
) -> WindowRankingResult:
    return WindowRankingResult(
        window_id=window_id,
        split="test",
        baseline=BaselineName.MAX_ANOMALY,
        node_scores=[
            NodeScore(
                node_id="a",
                score=1.0,
            ),
            NodeScore(
                node_id="b",
                score=0.5,
            ),
        ],
        ranked_node_ids=["a", "b"],
        true_root_cause_node_id=true_node,
        true_root_cause_rank=rank,
        abstained=abstained,
        top_score=1.0,
        score_margin=0.5,
        runtime_ms=1.0,
    )


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(2) == 0.5


def test_hits_at_k() -> None:
    assert hits_at_k(
        2,
        k=3,
    ) == 1.0

    assert hits_at_k(
        4,
        k=3,
    ) == 0.0


def test_summary_handles_faulty_and_healthy() -> None:
    results = [
        make_result(
            window_id="fault_1",
            true_node="a",
            rank=1,
            abstained=False,
        ),
        make_result(
            window_id="fault_2",
            true_node="b",
            rank=None,
            abstained=True,
        ),
        make_result(
            window_id="healthy_1",
            true_node=None,
            rank=None,
            abstained=True,
        ),
        make_result(
            window_id="healthy_2",
            true_node=None,
            rank=None,
            abstained=False,
        ),
    ]

    summary = summarize_ranking_results(
        results
    )

    assert summary.mrr == 0.5
    assert summary.faulty_coverage == 0.5
    assert summary.selective_mrr == 1.0
    assert summary.healthy_abstention_rate == 0.5
    assert (
        summary.healthy_false_selection_rate
        == 0.5
    )
