from datetime import datetime, timedelta, timezone

from app.domain.nrim.baselines.dual_path_episode_gate import (
    DualPathConfig,
    DualPathEpisodeGate,
    HealthyResidualModel,
    confirm_observable_impact,
    fit_healthy_residual_model,
    stage2_eligible_decision,
)
from app.domain.nrim.baselines.temporal_episode_gate import (
    EpisodeRecord,
    GateState,
    SupportAssessment,
    SupportStatus,
)


FEATURE_NAMES = [
    "node_type__catchup_service",
    "recent_change_event_count",
    "iptv__session__active_count__applicable",
    "iptv__session__active_count__missing",
    "iptv__session__active_count__latest",
    "history__iptv__session__active_count__robust_z",
    "history__iptv__session__active_count__delta",
    "iptv__catchup__service_availability__applicable",
    "iptv__catchup__service_availability__missing",
    "iptv__catchup__service_availability__latest",
    "history__iptv__catchup__service_availability__robust_z",
    "iptv__catchup__recording_failures__applicable",
    "iptv__catchup__recording_failures__missing",
    "iptv__catchup__recording_failures__latest",
    "history__iptv__catchup__recording_failures__robust_z",
]


def make_window(
    *,
    window_id: str = "window_1",
    cutoff_hour: int = 6,
    impact: bool = True,
) -> dict:
    values = {
        "node_type__catchup_service": 1.0,
        "recent_change_event_count": 0.0,
        "iptv__session__active_count__applicable": 1.0,
        "iptv__session__active_count__missing": 0.0,
        "iptv__session__active_count__latest": 5.0,
        "history__iptv__session__active_count__robust_z": (
            -3.0 if impact else 0.0
        ),
        "history__iptv__session__active_count__delta": (
            -10.0 if impact else 0.0
        ),
        "iptv__catchup__service_availability__applicable": 1.0,
        "iptv__catchup__service_availability__missing": 0.0,
        "iptv__catchup__service_availability__latest": (
            60.0 if impact else 100.0
        ),
        "history__iptv__catchup__service_availability__robust_z": (
            -3.0 if impact else 0.0
        ),
        "iptv__catchup__recording_failures__applicable": 1.0,
        "iptv__catchup__recording_failures__missing": 0.0,
        "iptv__catchup__recording_failures__latest": (
            3.0 if impact else 0.0
        ),
        "history__iptv__catchup__recording_failures__robust_z": (
            3.0 if impact else 0.0
        ),
    }
    cutoff = datetime(
        2026,
        1,
        1,
        cutoff_hour,
        tzinfo=timezone.utc,
    )
    return {
        "window_id": window_id,
        "source_scenario_id": "scenario_1",
        "split": "validation",
        "observation_start": (
            cutoff - timedelta(hours=6)
        ).isoformat(),
        "observation_cutoff": cutoff.isoformat(),
        "prediction_end": (
            cutoff + timedelta(hours=2)
        ).isoformat(),
        "node_ids": ["catchup_service_1"],
        "node_feature_names": FEATURE_NAMES,
        "node_features": [
            [values[name] for name in FEATURE_NAMES]
        ],
        "edge_index": [],
        "edge_feature_names": [],
        "edge_features": [],
        "targets": {
            "current_incident": int(impact),
        },
    }


def make_record(
    windows: tuple[dict, ...],
    *,
    healthy: bool = False,
) -> EpisodeRecord:
    return EpisodeRecord(
        scenario_id="scenario_1",
        split="validation",
        topology_group="topology_1",
        ordered_window_ids=tuple(
            window["window_id"]
            for window in windows
        ),
        windows=windows,
        impact_onset=None,
        recovery_timestamp=None,
        confounder_family="none",
        healthy_control=healthy,
        failure_family=(
            "healthy"
            if healthy
            else "catchup_worker_failure"
        ),
    )


def zero_residual_model() -> HealthyResidualModel:
    return HealthyResidualModel(
        feature_names=(),
        means=(),
        scales=(),
        weights=(),
        bias=0.0,
        residual_center=0.0,
        residual_scale=1.0,
    )


def metadata() -> dict:
    return {
        "room_count": 100,
        "floor_count": 4,
        "retention_days": 7,
        "base_occupancy_fraction": 0.6,
        "catchup_recording_channels": 30,
        "average_bitrate_mbps": 4.0,
        "shared_storage": True,
        "redundant_middleware": False,
    }


def test_impact_confirmation_requires_independent_signal_families() -> None:
    impact = confirm_observable_impact(
        window=make_window(),
    )

    assert impact.confirmed
    assert "service_traffic_decline" in impact.families
    assert (
        "service_availability_decline"
        in impact.families
    )
    assert "recording_errors" in impact.families
    assert impact.causal_consistent


def test_fast_path_opens_only_with_impact_confirmation() -> None:
    record = make_record((make_window(),))
    gate = DualPathEpisodeGate(
        config=DualPathConfig(
            fast_probability_threshold=0.80,
            fast_residual_threshold=1.0,
            enable_slow_path=False,
        ),
        residual_model=zero_residual_model(),
    )

    result = gate.run(
        record=record,
        probabilities=[0.90],
        metadata=metadata(),
    )

    assert result.decisions[0].state is GateState.INCIDENT
    assert result.decisions[0].activation_path == "fast"
    assert result.decisions[0].fast_triggered
    assert stage2_eligible_decision(
        result.decisions[0]
    )


def test_probability_alone_never_opens_default_fast_path() -> None:
    record = make_record(
        (make_window(impact=False),),
        healthy=True,
    )
    gate = DualPathEpisodeGate(
        config=DualPathConfig(
            fast_probability_threshold=0.80,
            fast_residual_threshold=1.0,
            enable_slow_path=False,
        ),
        residual_model=zero_residual_model(),
    )

    result = gate.run(
        record=record,
        probabilities=[0.90],
        metadata=metadata(),
    )

    assert result.decisions[0].state is GateState.SUSPECT
    assert not result.decisions[0].fast_triggered


def test_unsupported_windows_are_safe_and_never_enter_stage2() -> None:
    record = make_record((make_window(),))
    gate = DualPathEpisodeGate(
        config=DualPathConfig(),
        residual_model=zero_residual_model(),
    )

    result = gate.run(
        record=record,
        probabilities=[0.70],
        metadata=metadata(),
        support=[
            SupportAssessment(
                status=SupportStatus.UNSUPPORTED,
                support_score=2.0,
            )
        ],
    )

    assert result.decisions[0].state is GateState.UNKNOWN
    assert result.decisions[0].activation_path == "unsupported"
    assert not stage2_eligible_decision(
        result.decisions[0]
    )


def test_residual_model_uses_only_observable_operational_context() -> None:
    windows = (
        make_window(
            window_id="window_1",
            cutoff_hour=6,
            impact=False,
        ),
        make_window(
            window_id="window_2",
            cutoff_hour=8,
            impact=False,
        ),
    )
    record = make_record(windows, healthy=True)
    unsafe_metadata = {
        **metadata(),
        "environment_id": "forbidden",
        "pair_id": "forbidden",
        "confounders": ["forbidden"],
        "failure_type": "healthy",
    }

    model = fit_healthy_residual_model(
        training_records=[record],
        probabilities={"scenario_1": [0.20, 0.25]},
        scenario_metadata={
            "scenario_1": unsafe_metadata
        },
    )

    assert model.feature_names
    assert all(
        token not in name
        for name in model.feature_names
        for token in (
            "environment",
            "pair_id",
            "confounder",
            "failure_type",
        )
    )
