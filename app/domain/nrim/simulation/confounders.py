from __future__ import annotations

import random
from copy import deepcopy
from datetime import datetime
from typing import Any

from .scenario_design import ConfounderType


def apply_environment_confounders(
    *,
    environment_values: dict[str, Any],
    confounders: list[ConfounderType],
    rng: random.Random,
) -> dict[str, Any]:
    updated = deepcopy(environment_values)

    if ConfounderType.HIGH_HEALTHY_WORKLOAD in confounders:
        updated["base_occupancy_fraction"] = min(
            0.98,
            max(
                float(
                    updated["base_occupancy_fraction"]
                ),
                rng.uniform(0.80, 0.95),
            ),
        )

        updated["evening_peak_multiplier"] = max(
            float(updated["evening_peak_multiplier"]),
            rng.uniform(1.35, 1.65),
        )

    if ConfounderType.RETENTION_INCREASE in confounders:
        updated["retention_days"] = min(
            21,
            int(updated["retention_days"])
            + rng.choice([2, 3, 5]),
        )

    return updated


def confounder_event_records(
    *,
    confounders: list[ConfounderType],
    start_time: datetime,
    duration_hours: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for index, confounder in enumerate(confounders):
        event_fraction = rng.uniform(0.15, 0.65)

        event_time = start_time.timestamp() + (
            duration_hours * 3600.0 * event_fraction
        )

        timestamp = datetime.fromtimestamp(
            event_time,
            tz=start_time.tzinfo,
        )

        record = {
            "event_id": f"confounder_{index}",
            "confounder_type": confounder.value,
            "observed_at": timestamp.isoformat(),
            "model_visible": True,
        }

        if confounder == ConfounderType.RETENTION_INCREASE:
            record.update(
                {
                    "signal_name": (
                        "iptv.catchup."
                        "retention_policy_change"
                    ),
                    "statement": (
                        "CatchUP retention configuration changed."
                    ),
                }
            )

        elif (
            confounder
            == ConfounderType.TEMPORARY_LATENCY_SPIKE
        ):
            record.update(
                {
                    "signal_name": (
                        "system.disk.io_latency"
                    ),
                    "statement": (
                        "A temporary storage-latency increase "
                        "was observed."
                    ),
                }
            )

        elif (
            confounder
            == ConfounderType.HARMLESS_WORKER_RESTART
        ):
            record.update(
                {
                    "signal_name": (
                        "system.process.restart_count"
                    ),
                    "statement": (
                        "A CatchUP worker restart was observed."
                    ),
                }
            )

        elif (
            confounder
            == ConfounderType.TRANSIENT_RECORDING_ERRORS
        ):
            record.update(
                {
                    "signal_name": (
                        "iptv.catchup.recording_failures"
                    ),
                    "statement": (
                        "A short recording-error burst occurred."
                    ),
                }
            )

        elif (
            confounder
            == ConfounderType.HIGH_HEALTHY_WORKLOAD
        ):
            record.update(
                {
                    "signal_name": (
                        "iptv.session.active_count"
                    ),
                    "statement": (
                        "Elevated but valid workload was observed."
                    ),
                }
            )

        else:
            continue

        records.append(record)

    return records


def apply_observable_confounders(
    *,
    telemetry: list[dict[str, Any]],
    confounders: list[ConfounderType],
    rng: random.Random,
) -> list[dict[str, Any]]:
    updated = deepcopy(telemetry)

    if (
        ConfounderType.TEMPORARY_LATENCY_SPIKE
        in confounders
    ):
        candidates = [
            event
            for event in updated
            if event.get("signal_name")
            == "system.disk.io_latency"
        ]

        if candidates:
            selected_count = min(
                len(candidates),
                rng.randint(1, 3),
            )

            for event in rng.sample(
                candidates,
                k=selected_count,
            ):
                if isinstance(
                    event.get("value"),
                    (int, float),
                ):
                    event["value"] = round(
                        float(event["value"])
                        * rng.uniform(1.8, 3.0),
                        3,
                    )

                    attributes = event.setdefault(
                        "attributes",
                        [],
                    )

                    attributes.append(
                        {
                            "key": "simulated_transient",
                            "value": True,
                            "quality": "high",
                        }
                    )

    if (
        ConfounderType.TRANSIENT_RECORDING_ERRORS
        in confounders
    ):
        candidates = [
            event
            for event in updated
            if event.get("signal_name")
            == "iptv.catchup.recording_failures"
        ]

        if candidates:
            event = rng.choice(candidates)

            current = int(event.get("value", 0))
            event["value"] = current + rng.randint(1, 3)

    return updated
