from app.domain.nrim.baselines.diagnostic_experiments import (
    _bootstrap_clustered_mean,
    _classification_metrics,
    _ranking_summary,
)


def test_classification_metrics_report_operational_rates() -> None:
    metrics = _classification_metrics(
        [
            (0.9, 1),
            (0.8, 0),
            (0.2, 0),
            (0.1, 1),
        ],
        threshold=0.5,
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["healthy_false_selection_rate"] == 0.5


def test_ranking_summary_is_independent_of_detector() -> None:
    summary = _ranking_summary([1, 2, 4])

    assert summary["window_count"] == 3
    assert summary["hits_at_1"] == 1 / 3
    assert summary["hits_at_3"] == 2 / 3


def test_clustered_bootstrap_is_reproducible() -> None:
    clusters = {
        "topology_a": [0.1, 0.2],
        "topology_b": [-0.1, 0.0],
    }

    first = _bootstrap_clustered_mean(
        values_by_cluster=clusters,
        seed=42,
        repetitions=100,
    )
    second = _bootstrap_clustered_mean(
        values_by_cluster=clusters,
        seed=42,
        repetitions=100,
    )

    assert first == second
