from __future__ import annotations

from copy import deepcopy

from app.domain.nrim.baselines.shift_sentinels import (
    assess_support,
    fit_shift_sentinels,
)
from app.domain.nrim.baselines.temporal_episode_gate import (
    EpisodeRecord,
    SupportStatus,
)


FEATURE_NAMES = [
    "node_type__catchup_service",
    "in_degree",
    "out_degree",
    "recent_change_event_count",
]
for prefix in (
    "system__disk__utilization",
    "system__disk__io_latency",
    "system__disk__io_errors",
    "iptv__catchup__recording_failures",
    "system__process__restart_count",
    "iptv__session__active_count",
):
    FEATURE_NAMES.extend(
        [
            f"{prefix}__latest",
            f"{prefix}__mean",
            f"{prefix}__maximum",
            f"{prefix}__slope",
            f"{prefix}__standard_deviation",
            f"{prefix}__applicable",
            f"{prefix}__missing",
            f"history__{prefix}__robust_z",
            f"history__{prefix}__delta",
            f"history__{prefix}__persistence_count",
        ]
    )


def make_window(scenario_id: str, active_sessions: float = 20.0) -> dict:
    row = [0.0] * len(FEATURE_NAMES)
    row[FEATURE_NAMES.index("node_type__catchup_service")] = 1.0
    row[FEATURE_NAMES.index("out_degree")] = 1.0
    for prefix in (
        "system__disk__utilization",
        "system__disk__io_latency",
        "system__disk__io_errors",
        "iptv__catchup__recording_failures",
        "system__process__restart_count",
        "iptv__session__active_count",
    ):
        row[FEATURE_NAMES.index(f"{prefix}__applicable")] = 1.0
    row[
        FEATURE_NAMES.index("iptv__session__active_count__mean")
    ] = active_sessions
    return {
        "window_id": f"window_{scenario_id}",
        "source_scenario_id": scenario_id,
        "split": "train",
        "observation_start": "2026-01-01T00:00:00+00:00",
        "observation_cutoff": "2026-01-01T06:00:00+00:00",
        "node_ids": ["service"],
        "node_feature_names": FEATURE_NAMES,
        "node_features": [row],
        "edge_index": [],
        "edge_features": [],
        "targets": {
            "current_incident": 0,
            "root_cause_node": [0],
        },
    }


def make_record(scenario_id: str, active_sessions: float = 20.0) -> EpisodeRecord:
    window = make_window(scenario_id, active_sessions)
    return EpisodeRecord(
        scenario_id=scenario_id,
        split="train",
        topology_group="topology_a",
        ordered_window_ids=(window["window_id"],),
        windows=(window,),
        impact_onset=None,
        recovery_timestamp=None,
        confounder_family="none",
        healthy_control=True,
    )


def make_metadata(scenario_id: str, rooms: int = 100) -> dict:
    return {
        "scenario_id": scenario_id,
        "room_count": rooms,
        "floor_count": 5,
    }


def test_shift_sentinel_marks_far_workload_as_unsupported() -> None:
    records = [
        make_record(f"scenario_{index}", 18.0 + index)
        for index in range(5)
    ]
    metadata = {
        record.scenario_id: make_metadata(record.scenario_id, 100 + index)
        for index, record in enumerate(records)
    }
    model = fit_shift_sentinels(
        training_records=records,
        scenario_metadata=metadata,
    )
    shifted = deepcopy(records[0].windows[0])

    assessment = assess_support(
        model=model,
        window=shifted,
        metadata=make_metadata("shifted", rooms=10_000),
    )

    assert assessment.status is SupportStatus.UNSUPPORTED
    assert dict(assessment.axis_scores)["workload"] > 1.0
