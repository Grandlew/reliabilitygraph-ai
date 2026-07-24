from collections import defaultdict

from app.domain.nrim.simulation.model_ready_exporter import (
    add_causal_history_features,
)


def test_history_features_use_prior_windows_only() -> None:
    names = [
        "system__disk__io_errors__latest",
        "system__disk__io_errors__applicable",
        "system__disk__io_errors__missing",
        "history__system__disk__io_errors__robust_z",
        "history__system__disk__io_errors__delta",
        "history__system__disk__io_errors__persistence_count",
    ]
    history = defaultdict(list)
    persistence = defaultdict(int)
    first = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    second = [[5.0, 1.0, 0.0, 0.0, 0.0, 0.0]]

    add_causal_history_features(
        node_ids=["storage"],
        feature_names=names,
        node_features=first,
        history=history,
        persistence=persistence,
    )
    add_causal_history_features(
        node_ids=["storage"],
        feature_names=names,
        node_features=second,
        history=history,
        persistence=persistence,
    )

    assert first[0][3:] == [0.0, 0.0, 0.0]
    assert second[0][3] > 0.0
    assert second[0][4] == 5.0
    assert second[0][5] == 1.0
