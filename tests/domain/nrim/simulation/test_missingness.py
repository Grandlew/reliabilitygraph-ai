import random
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from app.domain.nrim.simulation.missingness import (
    apply_missingness,
)
from app.domain.nrim.simulation.scenario_design import (
    MissingnessMode,
    MissingnessPlan,
)


def make_telemetry(
    count: int = 100,
) -> list[dict]:
    start = datetime(
        2026,
        7,
        18,
        0,
        0,
        tzinfo=timezone.utc,
    )

    return [
        {
            "event_id": f"event_{index}",
            "signal_name": (
                "system.disk.io_latency"
                if index % 2 == 0
                else "system.disk.utilization"
            ),
            "observed_at": (
                start + timedelta(minutes=index)
            ).isoformat(),
            "value": float(index),
        }
        for index in range(count)
    ]


def test_no_missingness_preserves_every_event() -> None:
    telemetry = make_telemetry()

    retained, report = apply_missingness(
        telemetry=telemetry,
        plan=MissingnessPlan(),
        rng=random.Random(42),
    )

    assert retained == telemetry
    assert report["removed_count"] == 0


def test_independent_missingness_removes_events() -> None:
    telemetry = make_telemetry()

    retained, report = apply_missingness(
        telemetry=telemetry,
        plan=MissingnessPlan(
            mode=MissingnessMode.INDEPENDENT,
            probability=0.25,
        ),
        rng=random.Random(42),
    )

    assert len(retained) < len(telemetry)
    assert report["removed_count"] > 0


def test_source_specific_missingness_preserves_other_signal() -> None:
    telemetry = make_telemetry()

    retained, _ = apply_missingness(
        telemetry=telemetry,
        plan=MissingnessPlan(
            mode=(
                MissingnessMode.SOURCE_SPECIFIC
            ),
            probability=0.80,
            affected_signal_names=[
                "system.disk.io_latency"
            ],
        ),
        rng=random.Random(42),
    )

    original_utilization_count = sum(
        1
        for event in telemetry
        if event["signal_name"]
        == "system.disk.utilization"
    )

    retained_utilization_count = sum(
        1
        for event in retained
        if event["signal_name"]
        == "system.disk.utilization"
    )

    assert (
        retained_utilization_count
        == original_utilization_count
    )


def test_contiguous_outage_removes_time_block() -> None:
    telemetry = make_telemetry()

    retained, report = apply_missingness(
        telemetry=telemetry,
        plan=MissingnessPlan(
            mode=(
                MissingnessMode.CONTIGUOUS_OUTAGE
            ),
            outage_start_fraction=0.40,
            outage_duration_fraction=0.10,
        ),
        rng=random.Random(42),
    )

    assert len(retained) < len(telemetry)
    assert report["removed_count"] > 0


def test_missingness_is_reproducible() -> None:
    telemetry = make_telemetry()

    plan = MissingnessPlan(
        mode=MissingnessMode.INDEPENDENT,
        probability=0.20,
    )

    first, _ = apply_missingness(
        telemetry=telemetry,
        plan=plan,
        rng=random.Random(42),
    )

    second, _ = apply_missingness(
        telemetry=telemetry,
        plan=plan,
        rng=random.Random(42),
    )

    assert first == second
