from app.domain.nrim.baselines.baseline_evaluator import (
    choose_abstention_threshold,
    evaluate_split,
    evaluate_window,
)
from app.domain.nrim.baselines.models import (
    BaselineName,
)


def make_window(
    *,
    window_id: str,
    healthy: bool,
    storage_error: float,
) -> dict:
    return {
        "window_id": window_id,
        "split": "validation",
        "node_ids": [
            "storage",
            "client",
        ],
        "node_feature_names": [
            "node_type__catchup_storage",
            "node_type__smart_tv_group",
            "system__disk__io_errors__latest",
            "system__disk__io_errors__maximum",
        ],
        "node_features": [
            [
                1.0,
                0.0,
                storage_error,
                storage_error,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
        ],
        "edge_index": [],
        "edge_feature_names": [],
        "edge_features": [],
        "targets": {
            "root_cause_node": (
                [0, 0]
                if healthy
                else [1, 0]
            ),
            "affected_service_node": [0, 0],
            "failure_type": (
                "healthy"
                if healthy
                else "storage_io_degradation"
            ),
            "current_incident": 0,
            "future_incident": int(
                not healthy
            ),
        },
    }


def test_evaluate_window_finds_storage() -> None:
    result = evaluate_window(
        window=make_window(
            window_id="window_1",
            healthy=False,
            storage_error=20.0,
        ),
        baseline=BaselineName.ERROR_EVIDENCE,
        minimum_top_score=0.0,
    )

    assert (
        result.true_root_cause_rank
        == 1
    )


def test_threshold_selection_uses_constraints() -> None:
    windows = [
        make_window(
            window_id="healthy_1",
            healthy=True,
            storage_error=0.0,
        ),
        make_window(
            window_id="healthy_2",
            healthy=True,
            storage_error=0.0,
        ),
        make_window(
            window_id="fault_1",
            healthy=False,
            storage_error=20.0,
        ),
        make_window(
            window_id="fault_2",
            healthy=False,
            storage_error=15.0,
        ),
    ]

    threshold = choose_abstention_threshold(
        validation_windows=windows,
        baseline=BaselineName.ERROR_EVIDENCE,
        maximum_healthy_false_selection_rate=0.0,
        minimum_faulty_coverage=1.0,
    )

    assert threshold > 0.0


def test_none_threshold_disables_abstention() -> None:
    windows = [
        make_window(
            window_id="healthy_1",
            healthy=True,
            storage_error=0.0,
        ),
        make_window(
            window_id="fault_1",
            healthy=False,
            storage_error=20.0,
        ),
    ]

    results, summary = evaluate_split(
        windows=windows,
        baseline=BaselineName.ERROR_EVIDENCE,
        abstention_threshold=None,
    )

    assert all(not result.abstained for result in results)
    assert summary.faulty_coverage == 1.0
    assert summary.healthy_false_selection_rate == 1.0
