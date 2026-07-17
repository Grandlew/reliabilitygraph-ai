from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .dataset_builder import (
    build_scenario,
    sample_injection_time,
    save_scenario,
)
from .models import (
    FailureType,
    FaultSpecification,
    OperatingRegime,
)
from .quality_audit import (
    audit_observable_scenario,
    audit_root_cause_labels,
)
from .topology_generator import (
    generate_hotel_topology,
)


def main() -> None:
    seed = 42

    topology = generate_hotel_topology(
        room_count=180,
        floor_count=6,
        redundant_middleware=False,
        shared_storage=True,
        seed=seed,
    )

    start_time = datetime(
        2026,
        7,
        16,
        0,
        0,
        tzinfo=timezone.utc,
    )

    regime = OperatingRegime(
        duration_hours=36,
        sampling_interval_minutes=15,
        base_occupancy_fraction=0.65,
        evening_peak_multiplier=1.35,
        weekend_multiplier=1.10,
        catchup_recording_channels=40,
        average_bitrate_mbps=5.0,
        retention_days=7,
        random_seed=seed,
    )

    fault = FaultSpecification(
        failure_type=(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        target_node_id="catchup_storage_1",
        injection_time=sample_injection_time(
            start_time=start_time,
            end_time=start_time + timedelta(
                hours=regime.duration_hours
            ),
            sampling_interval_minutes=(
                regime.sampling_interval_minutes
            ),
            rng=random.Random(
                f"nrim-example:{seed}:injection"
            ),
        ),
        severity=0.75,
        parameters={
            "latency_multiplier": 4.0,
            "write_success_factor": 0.60,
        },
        expected_affected_service_types=[],
    )

    faulty = build_scenario(
        scenario_id="scenario_83f14c",
        topology=topology,
        regime=regime,
        start_time=start_time,
        fault=fault,
        randomize_injection_time=False,
    )

    healthy = build_scenario(
        scenario_id="scenario_92a7de",
        topology=topology,
        regime=regime,
        start_time=start_time,
        fault=None,
    )

    output_dir = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "simulation"
        / "generated"
    )

    save_scenario(
        scenario=faulty,
        output_dir=output_dir,
        scenario_id="scenario_83f14c",
    )

    save_scenario(
        scenario=healthy,
        output_dir=output_dir,
        scenario_id="scenario_92a7de",
    )

    for name, scenario in (
        ("faulty", faulty.observable),
        ("healthy", healthy.observable),
    ):
        leakage = audit_observable_scenario(
            scenario
        )
        integrity = audit_root_cause_labels(
            scenario
        )

        print(f"\n{name.upper()} SCENARIO")
        print(
            "Telemetry events:",
            len(scenario["telemetry"]),
        )
        print("Leakage warnings:", leakage)
        print("Integrity errors:", integrity)

    print("\nSaved scenarios to:", output_dir)


if __name__ == "__main__":
    main()
