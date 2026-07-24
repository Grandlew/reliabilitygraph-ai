from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.baselines.temporal_episode_gate import (
    EpisodeRecord,
    GateState,
    SupportAssessment,
    SupportStatus,
    TemporalEpisodeGate,
    TemporalGateConfig,
    assemble_episode_records,
    exact_binomial_upper_bound,
    summarize_episode_results,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def window(
    index: int,
    *,
    scenario_id: str = "scenario_a",
    split: str = "validation",
    incident: int = 0,
) -> dict:
    cutoff = START + timedelta(hours=index + 6)
    return {
        "window_id": f"window_{scenario_id}_{index}",
        "source_scenario_id": scenario_id,
        "split": split,
        "observation_start": (cutoff - timedelta(hours=6)).isoformat(),
        "observation_cutoff": cutoff.isoformat(),
        "targets": {"current_incident": incident},
    }


def metadata(
    scenario_id: str,
    *,
    split: str = "validation",
    topology: str = "topology_a",
    rooms: int = 100,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "split": split,
        "topology_fingerprint": topology,
        "room_count": rooms,
        "floor_count": 5,
        "missingness_mode": "source_specific",
        "missing_fraction": 0.02,
        "confounders": [],
        "healthy": True,
        "failure_type": "healthy",
    }


def record(truth: tuple[int, ...] = (0, 0, 1, 1, 0, 0)) -> EpisodeRecord:
    windows = tuple(
        window(index, incident=value)
        for index, value in enumerate(truth)
    )
    return EpisodeRecord(
        scenario_id="scenario_a",
        split="validation",
        topology_group="topology_a",
        ordered_window_ids=tuple(item["window_id"] for item in windows),
        windows=windows,
        impact_onset=START + timedelta(hours=8),
        recovery_timestamp=START + timedelta(hours=10),
        confounder_family="none",
        healthy_control=not any(truth),
    )


def test_sequence_assembler_sorts_and_audits_overlaps() -> None:
    train = metadata("scenario_train", split="train")
    validation = metadata("scenario_a")
    windows = [
        window(1),
        window(0),
        window(0, scenario_id="scenario_train", split="train"),
    ]

    records, audit = assemble_episode_records(
        windows=windows,
        scenario_metadata={
            "scenario_train": train,
            "scenario_a": validation,
        },
    )

    validation_record = next(
        item for item in records if item.scenario_id == "scenario_a"
    )
    assert validation_record.ordered_window_ids == (
        "window_scenario_a_0",
        "window_scenario_a_1",
    )
    assert audit.overlapping_window_pairs == 1
    assert audit.leakage_free


def test_sequence_assembler_rejects_cross_split_scenario() -> None:
    windows = [
        window(0),
        window(1, split="test"),
        window(0, scenario_id="scenario_train", split="train"),
    ]

    with pytest.raises(ValueError, match="cross_split=1"):
        assemble_episode_records(
            windows=windows,
            scenario_metadata={
                "scenario_train": metadata(
                    "scenario_train", split="train"
                ),
                "scenario_a": metadata("scenario_a"),
            },
        )


def test_duration_gate_rejects_one_window_spike() -> None:
    gate = TemporalEpisodeGate(
        TemporalGateConfig(
            window_threshold=0.5,
            evidence_center=0.4,
            decay=0.5,
            entry_threshold=0.1,
            exit_threshold=0.05,
            minimum_suspect_windows=2,
            maximum_suspect_windows=3,
            severity_override_probability=0.99,
        )
    )

    result = gate.run(
        record=record((0, 0, 0)),
        probabilities=(0.1, 0.8, 0.1),
    )

    assert GateState.INCIDENT not in {
        decision.state for decision in result.decisions
    }


def test_unsupported_semantics_never_return_healthy() -> None:
    gate = TemporalEpisodeGate(TemporalGateConfig())
    unsupported = [
        SupportAssessment(
            status=SupportStatus.UNSUPPORTED,
            support_score=2.0,
        )
    ] * 2

    result = gate.run(
        record=record((0, 0)),
        probabilities=(0.1, 0.9),
        support=unsupported,
    )

    assert [decision.state for decision in result.decisions] == [
        GateState.UNKNOWN,
        GateState.ESCALATE,
    ]


def test_exact_upper_bound_is_conservative_for_small_sample() -> None:
    bound = exact_binomial_upper_bound(
        events=0,
        trials=10,
        confidence=0.95,
    )

    assert bound.point_estimate == 0.0
    assert bound.upper_confidence_bound == pytest.approx(
        1.0 - 0.05 ** (1.0 / 10.0)
    )
    assert bound.upper_confidence_bound > 0.10


def test_episode_metrics_count_scenarios_not_windows() -> None:
    config = TemporalGateConfig(
        window_threshold=0.5,
        evidence_center=0.5,
        decay=0.0,
        entry_threshold=0.0,
        exit_threshold=0.0,
        minimum_suspect_windows=1,
        minimum_recovery_windows=1,
    )
    gate = TemporalEpisodeGate(config)
    faulty = gate.run(
        record=record((0, 0, 1, 1, 0)),
        probabilities=(0.1, 0.1, 0.9, 0.9, 0.1),
    )
    healthy_record = record((0, 0, 0, 0, 0))
    healthy = gate.run(
        record=healthy_record,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1),
    )

    summary = summarize_episode_results([faulty, healthy])

    assert summary["healthy_scenario_count"] == 1
    assert summary["true_episode_count"] == 1
    assert summary["incident_episode_recall"] == 1.0
