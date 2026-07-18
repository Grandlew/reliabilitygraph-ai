from __future__ import annotations

import random
from copy import deepcopy
from datetime import datetime
from typing import Any

from .scenario_design import (
    MissingnessMode,
    MissingnessPlan,
)


def parse_observed_at(
    event: dict[str, Any],
) -> datetime:
    raw_timestamp = event.get("observed_at")

    if not isinstance(raw_timestamp, str):
        raise ValueError(
            "Telemetry event has no ISO observed_at value."
        )

    return datetime.fromisoformat(
        raw_timestamp.replace("Z", "+00:00")
    )


def apply_missingness(
    *,
    telemetry: list[dict[str, Any]],
    plan: MissingnessPlan,
    rng: random.Random,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    original = deepcopy(telemetry)

    if not original:
        return [], {
            "original_count": 0,
            "retained_count": 0,
            "removed_count": 0,
            "removed_fraction": 0.0,
            "mode": plan.mode.value,
        }

    retained: list[dict[str, Any]] = []

    if plan.mode == MissingnessMode.NONE:
        retained = original

    elif plan.mode == MissingnessMode.INDEPENDENT:
        for event in original:
            if rng.random() >= plan.probability:
                retained.append(event)

    elif plan.mode == MissingnessMode.SOURCE_SPECIFIC:
        affected_names = set(
            plan.affected_signal_names
        )

        for event in original:
            is_affected = (
                event.get("signal_name")
                in affected_names
            )

            remove = (
                is_affected
                and rng.random() < plan.probability
            )

            if not remove:
                retained.append(event)

    elif (
        plan.mode
        == MissingnessMode.CONTIGUOUS_OUTAGE
    ):
        timestamps = [
            parse_observed_at(event)
            for event in original
        ]

        start_time = min(timestamps)
        end_time = max(timestamps)

        total_seconds = (
            end_time - start_time
        ).total_seconds()

        outage_start = start_time.timestamp() + (
            total_seconds
            * float(plan.outage_start_fraction)
        )

        outage_duration = (
            total_seconds
            * float(plan.outage_duration_fraction)
        )

        outage_end = outage_start + outage_duration

        for event in original:
            observed_seconds = parse_observed_at(
                event
            ).timestamp()

            if not (
                outage_start
                <= observed_seconds
                <= outage_end
            ):
                retained.append(event)

    else:
        raise ValueError(
            f"Unsupported missingness mode: "
            f"{plan.mode.value}"
        )

    removed_count = len(original) - len(retained)

    return retained, {
        "original_count": len(original),
        "retained_count": len(retained),
        "removed_count": removed_count,
        "removed_fraction": round(
            removed_count / len(original),
            6,
        ),
        "mode": plan.mode.value,
    }
