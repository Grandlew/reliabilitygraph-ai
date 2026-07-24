from app.domain.nrim.baselines.learned_fusion import (
    fit_root_cause_fusion,
    learned_fusion_scores,
)
from tests.domain.nrim.baselines.test_baseline_evaluator import (
    make_window,
)


def test_learned_fusion_ranks_observed_root_above_client() -> None:
    windows = [
        make_window(
            window_id=f"fault_{index}",
            healthy=False,
            storage_error=20.0 + index,
        )
        for index in range(4)
    ]
    model = fit_root_cause_fusion(
        training_windows=windows,
        iterations=100,
    )
    scores = learned_fusion_scores(
        window=windows[0],
        model=model,
    )
    by_id = {
        item.node_id: item.score
        for item in scores
    }

    assert by_id["storage"] > by_id["client"]


def test_post_freeze_stage1_signal_cannot_change_stage2_scores() -> None:
    windows = [
        make_window(
            window_id=f"fault_{index}",
            healthy=False,
            storage_error=20.0 + index,
        )
        for index in range(4)
    ]
    model = fit_root_cause_fusion(
        training_windows=windows,
        iterations=100,
    )
    original = learned_fusion_scores(
        window=windows[0],
        model=model,
    )
    augmented = dict(windows[0])
    augmented["node_feature_names"] = [
        *windows[0]["node_feature_names"],
        "iptv__catchup__service_availability__latest",
        "iptv__catchup__service_availability__mean",
        "iptv__catchup__service_availability__slope",
        (
            "history__iptv__catchup__service_availability"
            "__robust_z"
        ),
    ]
    augmented["node_features"] = [
        [*row, 1.0, 100.0, -99.0, -50.0]
        for row in windows[0]["node_features"]
    ]
    after = learned_fusion_scores(
        window=augmented,
        model=model,
    )

    assert [
        (item.node_id, item.score)
        for item in after
    ] == [
        (item.node_id, item.score)
        for item in original
    ]
