from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .telemetry import (
    CanonicalTelemetryEvent,
    TelemetryBatch,
    TelemetryQuality,
)


@dataclass
class TelemetryValidationResult:
    accepted: list[CanonicalTelemetryEvent] = field(default_factory=list)
    quarantined: list[CanonicalTelemetryEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_topology_mapping(
    event: CanonicalTelemetryEvent,
    *,
    known_node_ids: set[str],
) -> str | None:
    if event.component_node_id not in known_node_ids:
        return (
            f"Event {event.event_id} references unknown component "
            f"{event.component_node_id}."
        )

    unknown_services = [
        service_id
        for service_id in event.service_node_ids
        if service_id not in known_node_ids
    ]

    if unknown_services:
        return (
            f"Event {event.event_id} references unknown service nodes: "
            f"{unknown_services}"
        )

    return None


def validate_events(
    events: list[CanonicalTelemetryEvent],
    *,
    known_node_ids: set[str],
) -> TelemetryValidationResult:
    result = TelemetryValidationResult()
    seen_event_ids: set[str] = set()

    for event in events:
        if event.event_id in seen_event_ids:
            error_msg = f"Duplicate event ID: {event.event_id}"
            result.errors.append(error_msg)
            quarantined_event = event.model_copy(
                update={
                    "quality": TelemetryQuality.QUARANTINED,
                }
            )
            result.quarantined.append(quarantined_event)
            continue

        seen_event_ids.add(event.event_id)

        mapping_error = validate_topology_mapping(
            event,
            known_node_ids=known_node_ids,
        )

        if mapping_error:
            result.errors.append(mapping_error)

            quarantined_event = event.model_copy(
                update={
                    "quality": TelemetryQuality.QUARANTINED,
                }
            )
            result.quarantined.append(quarantined_event)
            continue

        result.accepted.append(event)

    return result


def load_telemetry_batch(path: Path) -> TelemetryBatch:
    if not path.exists():
        raise FileNotFoundError(f"Telemetry batch not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_batch = json.load(file)

    return TelemetryBatch.model_validate(raw_batch)


if __name__ == "__main__":
    # Note: These paths assume a specific file structure.
    # In a real environment, you would ensure these files exist.
    base_dir = Path("/content").resolve()
    example_dir = base_dir / "examples"

    for filename in (
        "hotel_180_rooms_telemetry.json",
        "hotel_180_rooms_changes.json",
    ):
        path = example_dir / filename
        if path.exists():
            batch = load_telemetry_batch(path)
            print(
                f"Valid telemetry batch: {batch.batch_id} "
                f"({len(batch.events)} events)"
            )
