import random

import pytest
from pydantic import ValidationError

from app.domain.nrim.simulation.models import (
    FailureType,
)
from app.domain.nrim.simulation.scenario_design import (
    MissingnessMode,
    MissingnessPlan,
    ScenarioPlan,
    opaque_identifier,
    sample_environment,
)


def test_opaque_identifier_is_reproducible() -> None:
    first = opaque_identifier(
        namespace="scenario",
        values={"seed": 42, "member": 1},
    )

    second = opaque_identifier(
        namespace="scenario",
        values={"member": 1, "seed": 42},
    )

    assert first == second


def test_identifier_does_not_expose_failure_name() -> None:
    identifier = opaque_identifier(
        namespace="scenario",
        values={
            "failure_type": (
                "storage_io_degradation"
            ),
            "seed": 42,
        },
    )

    assert "storage" not in identifier
    assert "degradation" not in identifier


def test_sample_environment_is_reproducible() -> None:
    first = sample_environment(
        rng=random.Random(42),
        environment_index=1,
    )

    second = sample_environment(
        rng=random.Random(42),
        environment_index=1,
    )

    assert first == second


def test_healthy_plan_rejects_fault_values() -> None:
    environment = sample_environment(
        rng=random.Random(42),
        environment_index=1,
    )

    with pytest.raises(ValidationError):
        ScenarioPlan(
            scenario_id="scenario_a",
            pair_id="pair_a",
            environment=environment,
            failure_type=FailureType.HEALTHY,
            fault_severity=0.5,
            fault_injection_fraction=0.5,
            topology_seed=1,
            workload_seed=2,
            observation_seed=3,
            fault_seed=4,
        )


def test_fault_plan_requires_severity() -> None:
    environment = sample_environment(
        rng=random.Random(42),
        environment_index=1,
    )

    with pytest.raises(ValidationError):
        ScenarioPlan(
            scenario_id="scenario_a",
            pair_id="pair_a",
            environment=environment,
            failure_type=(
                FailureType.STORAGE_IO_DEGRADATION
            ),
            fault_severity=None,
            fault_injection_fraction=0.5,
            topology_seed=1,
            workload_seed=2,
            observation_seed=3,
            fault_seed=4,
        )


def test_contiguous_missingness_requires_window() -> None:
    with pytest.raises(ValidationError):
        MissingnessPlan(
            mode=(
                MissingnessMode.CONTIGUOUS_OUTAGE
            ),
        )
