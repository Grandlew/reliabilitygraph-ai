from datetime import datetime, timezone

from app.domain.nrim.simulation.dataset_builder import (
    initialize_states,
)
from app.domain.nrim.simulation.models import (
    FailureType,
    FaultSpecification,
)
from app.domain.nrim.simulation.propagation import (
    propagate_fault,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)


def test_storage_fault_propagates_to_catchup_service() -> None:
    topology = generate_hotel_topology(
        room_count=100,
        floor_count=4,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )

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
        severity=0.9,
    )

    updated, records = propagate_fault(
        topology=topology,
        fault=fault,
        states=states,
        timestamp=now,
    )

    assert records
    assert any(
        record.target_node_id
        == "catchup_service_1"
        for record in records
    )

    assert (
        updated["catchup_service_1"]
        .latent_error_factor
        > 1.0
    )
