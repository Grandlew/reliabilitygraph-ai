import random
from datetime import datetime, timezone

from app.domain.nrim.simulation.models import (
    FailureType,
)
from app.domain.nrim.simulation.scenario_design import (
    ScenarioPlan,
    sample_environment,
)
from app.domain.nrim.simulation.scenario_factory import (
    build_planned_scenario,
)


def make_plan(
    failure_type: FailureType,
) -> ScenarioPlan:
    environment = sample_environment(
        rng=random.Random(42),
        environment_index=1,
    )

    return ScenarioPlan(
        scenario_id=(
            "scenario_fault"
            if failure_type != FailureType.HEALTHY
            else "scenario_healthy"
        ),
        pair_id="pair_1",
        environment=environment,
        failure_type=failure_type,
        fault_severity=(
            0.75
            if failure_type != FailureType.HEALTHY
            else None
        ),
        fault_injection_fraction=(
            0.50
            if failure_type != FailureType.HEALTHY
            else None
        ),
        topology_seed=1,
        workload_seed=2,
        observation_seed=3,
        fault_seed=4,
    )


def test_factory_builds_healthy_scenario() -> None:
    scenario, _ = build_planned_scenario(
        plan=make_plan(FailureType.HEALTHY),
        start_time=datetime.now(timezone.utc),
    )

    assert (
        scenario.observable["labels"][
            "failure_type"
        ]
        == "healthy"
    )


def test_factory_builds_fault_scenario() -> None:
    scenario, _ = build_planned_scenario(
        plan=make_plan(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        start_time=datetime.now(timezone.utc),
    )

    assert (
        scenario.observable["labels"][
            "failure_type"
        ]
        == "storage_io_degradation"
    )

    assert (
        sum(
            scenario.observable["labels"][
                "root_cause_node"
            ].values()
        )
        == 1
    )


def test_observable_does_not_contain_hidden_fault() -> None:
    scenario, _ = build_planned_scenario(
        plan=make_plan(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        start_time=datetime.now(timezone.utc),
    )

    observable_text = str(
        scenario.observable
    ).lower()

    assert "latent_latency_factor" not in observable_text
    assert "caused_by_fault_id" not in observable_text
    assert "fault_seed" not in observable_text
