import pytest

from app.domain.nrim.baselines.feature_access import (
    FeatureAccessor,
)
from app.domain.nrim.baselines.root_cause_baselines import (
    deterministic_random_score,
    rank_node_scores,
    robust_scale,
    static_criticality_baseline,
    weighted_feature_score,
)


def make_window() -> dict:
    return {
        "window_id": "window_abc",
        "node_ids": [
            "storage",
            "client",
        ],
        "node_feature_names": [
            "node_type__catchup_storage",
            "node_type__smart_tv_group",
            "system__disk__io_errors__maximum",
        ],
        "node_features": [
            [1.0, 0.0, 8.0],
            [0.0, 1.0, 0.0],
        ],
        "edge_index": [],
        "edge_feature_names": [],
        "edge_features": [],
    }


def test_random_score_is_reproducible() -> None:
    first = deterministic_random_score(
        window_id="window_1",
        node_id="node_1",
        seed=42,
    )

    second = deterministic_random_score(
        window_id="window_1",
        node_id="node_1",
        seed=42,
    )

    assert first == second


def test_storage_has_higher_static_criticality() -> None:
    scores = static_criticality_baseline(
        window=make_window()
    )

    by_id = {
        score.node_id: score.score
        for score in scores
    }

    assert by_id["storage"] > by_id["client"]


def test_robust_scale_is_bounded() -> None:
    assert robust_scale(
        0.0,
        scale=10.0,
    ) == 0.0

    assert 0.0 < robust_scale(
        100.0,
        scale=10.0,
    ) < 1.0


def test_weighted_feature_score_uses_evidence() -> None:
    accessor = FeatureAccessor(
        ["system__disk__io_errors__maximum"]
    )

    score, evidence = weighted_feature_score(
        row=[10.0],
        accessor=accessor,
        feature_weights={
            "system__disk__io_errors__maximum": 1.0
        },
    )

    assert score > 0.0
    assert evidence


def test_ranking_tie_breaks_by_node_id() -> None:
    from app.domain.nrim.baselines.models import (
        NodeScore,
    )

    ranked = rank_node_scores(
        [
            NodeScore(
                node_id="b",
                score=1.0,
            ),
            NodeScore(
                node_id="a",
                score=1.0,
            ),
        ]
    )

    assert [
        item.node_id
        for item in ranked
    ] == ["a", "b"]


def test_ranking_rejects_duplicate_nodes() -> None:
    from app.domain.nrim.baselines.models import (
        NodeScore,
    )

    with pytest.raises(ValueError):
        rank_node_scores(
            [
                NodeScore(
                    node_id="a",
                    score=1.0,
                ),
                NodeScore(
                    node_id="a",
                    score=0.5,
                ),
            ]
        )
