from datetime import datetime, timezone

import pytest

from app.domain.nrim.simulation.dataset_builder import (
    initialize_states,
)
from app.domain.nrim.simulation.fault_injector import (
    inject_fault,
    validate_fault_target,
)
from app.domain.nrim.simulation.models import (
    FailureType,
    FaultSpecification,
    HealthState,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)


def make_topology():
    return generate_hotel_topology(
        room_count=100,
        floor_count=4,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )


def test_storage_fault_rejects_middleware_target() -> None:
    topology = make_topology()

    fault = FaultSpecification(
        failure_type=(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        target_node_id="middleware_1",
        injection_time=datetime.now(timezone.utc),
        severity=0.8,
    )

    with pytest.raises(ValueError):
        validate_fault_target(
            topology=topology,
            fault=fault,
        )


def test_storage_fault_degrades_storage_state() -> None:
    topology = make_topology()
    now = datetime.now(timezone.utc)

    states = initialize_states(
        topology=topology,
        timestamp=now,
    )

    fault = FaultSpecification(
        failure_type=(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        target_node_id="catchup_storage_1",
        injection_time=now,
        severity=0.8,
    )

    updated = inject_fault(
        topology=topology,
        fault=fault,
        states=states,
    )

    assert (
        updated["catchup_storage_1"].health_state
        == HealthState.DEGRADED
    )
    assert (
        updated["catchup_storage_1"]
        .latent_latency_factor
        > 1.0
    )
