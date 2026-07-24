from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from .confounders import (
    apply_environment_confounders,
    apply_observable_confounders,
    confounder_event_records,
)
from .dataset_builder import (
    GeneratedScenario,
    build_scenario,
)
from .missingness import apply_missingness
from .models import (
    FailureType,
    FaultSpecification,
    OperatingRegime,
)
from .scenario_design import ScenarioPlan
from .topology_generator import (
    generate_hotel_topology,
)


def valid_fault_target(
    failure_type: FailureType,
) -> str:
    if failure_type in {
        FailureType.STORAGE_CAPACITY_SATURATION,
        FailureType.STORAGE_IO_DEGRADATION,
    }:
        return "catchup_storage_1"

    if failure_type in {
        FailureType.CLEANUP_JOB_FAILURE,
        FailureType.CATCHUP_WORKER_FAILURE,
    }:
        return "catchup_service_1"

    raise ValueError(
        f"No target mapping for failure type "
        f"{failure_type.value}."
    )


def build_regime(
    *,
    plan: ScenarioPlan,
) -> OperatingRegime:
    environment_values = (
        plan.environment.model_dump()
    )

    environment_rng = random.Random(
        plan.workload_seed
    )

    updated = apply_environment_confounders(
        environment_values=environment_values,
        confounders=plan.confounders,
        rng=environment_rng,
    )

    return OperatingRegime(
        duration_hours=int(
            updated["scenario_duration_hours"]
        ),
        sampling_interval_minutes=int(
            updated["sampling_interval_minutes"]
        ),
        base_occupancy_fraction=float(
            updated["base_occupancy_fraction"]
        ),
        evening_peak_multiplier=float(
            updated["evening_peak_multiplier"]
        ),
        weekend_multiplier=float(
            updated["weekend_multiplier"]
        ),
        catchup_recording_channels=int(
            updated["catchup_recording_channels"]
        ),
        average_bitrate_mbps=float(
            updated["average_bitrate_mbps"]
        ),
        retention_days=int(
            updated["retention_days"]
        ),
        random_seed=plan.workload_seed,
    )


def build_fault(
    *,
    plan: ScenarioPlan,
    start_time: datetime,
    regime: OperatingRegime,
) -> FaultSpecification | None:
    if plan.failure_type == FailureType.HEALTHY:
        return None

    duration = timedelta(
        hours=regime.duration_hours
    )

    injection_time = start_time + (
        duration
        * float(plan.fault_injection_fraction)
    )

    return FaultSpecification(
        failure_type=plan.failure_type,
        target_node_id=valid_fault_target(
            plan.failure_type
        ),
        injection_time=injection_time,
        severity=float(plan.fault_severity),
        parameters={
            "fault_seed": plan.fault_seed,
            "fault_duration_hours": (
                plan.fault_duration_hours
            ),
        },
    )


def build_planned_scenario(
    *,
    plan: ScenarioPlan,
    start_time: datetime,
) -> tuple[
    GeneratedScenario,
    dict[str, Any],
]:
    topology = generate_hotel_topology(
        room_count=plan.environment.room_count,
        floor_count=plan.environment.floor_count,
        redundant_middleware=(
            plan.environment.redundant_middleware
        ),
        shared_storage=(
            plan.environment.shared_storage
        ),
        seed=plan.topology_seed,
    )

    regime = build_regime(plan=plan)

    fault = build_fault(
        plan=plan,
        start_time=start_time,
        regime=regime,
    )

    scenario = build_scenario(
        scenario_id=plan.scenario_id,
        topology=topology,
        regime=regime,
        start_time=start_time,
        fault=fault,
    )

    observation_rng = random.Random(
        plan.observation_seed
    )

    observable = dict(scenario.observable)

    telemetry = apply_observable_confounders(
        telemetry=list(
            observable["telemetry"]
        ),
        confounders=plan.confounders,
        rng=observation_rng,
    )

    telemetry, missingness_report = (
        apply_missingness(
            telemetry=telemetry,
            plan=plan.missingness,
            rng=observation_rng,
        )
    )

    observable["telemetry"] = telemetry
    observable["environment_id"] = (
        plan.environment.environment_id
    )
    observable["pair_id"] = plan.pair_id

    observable["context_events"] = (
        confounder_event_records(
            confounders=plan.confounders,
            start_time=start_time,
            duration_hours=regime.duration_hours,
            rng=observation_rng,
        )
    )

    observable["simulation_metadata"] = {
        "confounder_count": len(
            plan.confounders
        ),
        "missingness_mode": (
            plan.missingness.mode.value
        ),
    }

    hidden = dict(scenario.hidden)

    hidden["scenario_plan"] = plan.model_dump(
        mode="json"
    )
    hidden["missingness_report"] = (
        missingness_report
    )

    return (
        GeneratedScenario(
            observable=observable,
            hidden=hidden,
        ),
        missingness_report,
    )
