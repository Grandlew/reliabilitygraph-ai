from app.domain.nrim.benchmark.models import (
    WindowSummary,
)
from app.domain.nrim.benchmark.shortcut_audit import (
    balanced_accuracy,
    run_shortcut_audit,
)


def make_summary(
    *,
    window_id: str,
    split: str,
    node_count: int,
    future_incident: int,
) -> WindowSummary:
    return WindowSummary(
        window_id=window_id,
        split=split,
        path="window.json",
        node_count=node_count,
        edge_count=node_count - 1,
        node_feature_count=20,
        edge_feature_count=8,
        failure_type=(
            "storage_io_degradation"
            if future_incident
            else "healthy"
        ),
        current_incident=0,
        future_incident=future_incident,
        root_cause_positive_count=(
            1 if future_incident else 0
        ),
        affected_service_positive_count=0,
        missing_feature_fraction=0.1,
    )


def test_balanced_accuracy() -> None:
    score = balanced_accuracy(
        targets=[0, 0, 1, 1],
        predictions=[0, 0, 1, 1],
    )

    assert score == 1.0


def test_shortcut_audit_detects_node_count_leakage() -> None:
    summaries = []

    for split in (
        "train",
        "validation",
    ):
        for index in range(20):
            target = index % 2

            summaries.append(
                make_summary(
                    window_id=(
                        f"{split}_{index}"
                    ),
                    split=split,
                    node_count=(
                        100 if target else 10
                    ),
                    future_incident=target,
                )
            )

    results = run_shortcut_audit(
        summaries
    )

    node_count_result = next(
        result
        for result in results
        if result.shortcut_name
        == "node_count"
    )

    assert node_count_result.suspicious is True
