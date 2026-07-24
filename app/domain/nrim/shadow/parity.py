from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.domain.nrim.simulation.sanitizer import (
    sanitize_observable_scenario,
)

from .contracts import (
    Applicability,
    ComponentType,
    DataQualityState,
    DeploymentProfile,
    DependencyType,
    DirectionSemantics,
    MetricName,
    MetricUnit,
    ObservationQuality,
    OperationalEvent,
    ServingSnapshot,
    SnapshotMode,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from .hashing import canonical_hash
from .privacy import Pseudonymizer
from .replay import ServingFeatureBuilder


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def synthetic_shadow_snapshot(
    *,
    observable_scenario: dict[str, Any],
    scenario_metadata: dict[str, Any],
    reference_window: dict[str, Any],
    pseudonymizer: Pseudonymizer,
) -> tuple[ServingSnapshot, dict[str, str]]:
    """Adapt a development-only scenario to the real serving contracts."""

    sanitized = sanitize_observable_scenario(observable_scenario)
    deployment = pseudonymizer.pseudonymize(
        str(sanitized["topology"]["topology_id"]),
        namespace="deployment",
    )
    node_map = {
        str(node["node_id"]): pseudonymizer.pseudonymize(
            str(node["node_id"]),
            namespace="component",
        )
        for node in sanitized["topology"]["nodes"]
    }
    node_types = {
        str(node["node_id"]): ComponentType(str(node["node_type"]))
        for node in sanitized["topology"]["nodes"]
    }
    start = _time(reference_window["observation_start"])
    cutoff = _time(reference_window["observation_cutoff"])
    version = "synthetic-parity-v1"
    components = tuple(
        TopologyComponent(
            deployment_pseudonym=deployment,
            component_pseudonym=node_map[str(node["node_id"])],
            component_type=ComponentType(str(node["node_type"])),
            valid_from_utc=start - timedelta(seconds=1),
            topology_version=version,
            source_system="v0.6-development-parity",
        )
        for node in sanitized["topology"]["nodes"]
    )
    direction = {
        "depends_on": DirectionSemantics.DESTINATION_DEPENDS_ON_SOURCE,
        "stores_on": DirectionSemantics.DESTINATION_DEPENDS_ON_SOURCE,
        "connected_to": DirectionSemantics.BIDIRECTIONAL,
    }
    edges = tuple(
        TopologyEdge(
            deployment_pseudonym=deployment,
            source_component=node_map[str(edge["source_node_id"])],
            destination_component=node_map[str(edge["target_node_id"])],
            dependency_type=DependencyType(str(edge["edge_type"])),
            direction_semantics=direction.get(
                str(edge["edge_type"]),
                DirectionSemantics.SOURCE_TO_DESTINATION,
            ),
            valid_from_utc=start - timedelta(seconds=1),
            topology_version=version,
            source_system="v0.6-development-parity",
            propagation_delay_minutes=int(
                edge.get("propagation_delay_minutes", 0)
            ),
            propagation_strength=float(
                edge.get("propagation_strength", 1.0)
            ),
        )
        for edge in sanitized["topology"]["edges"]
    )
    unit_map = {
        MetricName.DISK_UTILIZATION: MetricUnit.PERCENT,
        MetricName.DISK_IO_LATENCY: MetricUnit.MILLISECONDS,
        MetricName.DISK_IO_ERRORS: MetricUnit.ERRORS_PER_INTERVAL,
        MetricName.RECORDING_FAILURES: MetricUnit.FAILURES_PER_INTERVAL,
        MetricName.PROCESS_RESTART_COUNT: MetricUnit.RESTARTS_PER_INTERVAL,
        MetricName.ACTIVE_SESSION_COUNT: MetricUnit.SESSIONS,
        MetricName.SERVICE_AVAILABILITY: MetricUnit.PERCENT,
    }
    observations = []
    for event in sanitized.get("telemetry", []):
        event_time = _time(str(event["observed_at"]))
        if not start <= event_time <= cutoff:
            continue
        metric = MetricName(str(event["signal_name"]))
        node_id = str(event["component_node_id"])
        value = event.get("value")
        observations.append(
            TelemetryObservation(
                deployment_pseudonym=deployment,
                component_pseudonym=node_map[node_id],
                component_type=node_types[node_id],
                metric_name=metric,
                metric_value=(
                    float(value) if value is not None else None
                ),
                unit=unit_map[metric],
                applicability=(
                    Applicability.OBSERVED
                    if value is not None
                    else Applicability.MISSING
                ),
                quality=ObservationQuality(
                    str(event.get("quality", "medium"))
                ),
                event_time_utc=event_time,
                ingestion_time_utc=_time(
                    str(event.get("ingested_at", event["observed_at"]))
                ),
                collector_id=str(
                    event.get("collection_source", "development")
                ),
                source_sequence_id=str(event["event_id"]),
                topology_version=version,
            )
        )
    operational_events = []
    for event in sanitized.get("context_events", []):
        event_time = _time(str(event["observed_at"]))
        if not start <= event_time <= cutoff:
            continue
        component = event.get("component_node_id")
        operational_events.append(
            OperationalEvent(
                deployment_pseudonym=deployment,
                component_pseudonym=(
                    node_map[str(component)]
                    if component is not None
                    else None
                ),
                signal_name=str(event["signal_name"]),
                event_time_utc=event_time,
                ingestion_time_utc=event_time,
                source_sequence_id=str(
                    event.get(
                        "event_id",
                        "event_" + canonical_hash(event)[:20],
                    )
                ),
                topology_version=version,
            )
        )
    profile = DeploymentProfile(
        deployment_pseudonym=deployment,
        profile_version="synthetic-parity-profile",
        valid_from_utc=start - timedelta(seconds=1),
        software_version="simulator-v0.5.1",
        room_count=int(scenario_metadata["room_count"]),
        floor_count=int(scenario_metadata["floor_count"]),
        retention_days=int(scenario_metadata["retention_days"]),
        base_occupancy_fraction=float(
            scenario_metadata["base_occupancy_fraction"]
        ),
        catchup_recording_channels=int(
            scenario_metadata["catchup_recording_channels"]
        ),
        average_bitrate_mbps=float(
            scenario_metadata["average_bitrate_mbps"]
        ),
        shared_storage=bool(scenario_metadata["shared_storage"]),
        redundant_middleware=bool(
            scenario_metadata["redundant_middleware"]
        ),
        collector_family="nrim_simulator",
    )
    return (
        ServingSnapshot(
            snapshot_id=str(reference_window["window_id"]),
            deployment_pseudonym=deployment,
            observation_start_utc=start,
            decision_cutoff_utc=cutoff,
            as_of_ingestion_time_utc=cutoff,
            topology_version=version,
            profile=profile,
            components=components,
            edges=edges,
            observations=tuple(observations),
            operational_events=tuple(operational_events),
            data_quality_state=DataQualityState.ACCEPTED,
            expected_observation_count=len(observations),
            available_observation_count=len(observations),
            mode=SnapshotMode.GOLDEN_REPLAY,
        ),
        node_map,
    )


def normalized_reference_window(
    *,
    reference_window: dict[str, Any],
    node_map: dict[str, str],
) -> dict[str, Any]:
    allowed = (
        "window_id",
        "node_feature_names",
        "node_features",
        "edge_index",
        "edge_feature_names",
        "edge_features",
    )
    result = {
        name: reference_window[name]
        for name in allowed
    }
    result.update(
        {
            "observation_start": _time(
                reference_window["observation_start"]
            ).isoformat(),
            "observation_cutoff": _time(
                reference_window["observation_cutoff"]
            ).isoformat(),
            "node_ids": [
                node_map[str(item)]
                for item in reference_window["node_ids"]
            ],
        }
    )
    return result


def verify_synthetic_serving_parity(
    *,
    observable_scenario: dict[str, Any],
    scenario_metadata: dict[str, Any],
    reference_window: dict[str, Any],
    pseudonymization_secret: bytes,
) -> dict[str, Any]:
    pseudonymizer = Pseudonymizer(
        pseudonymization_secret,
        key_id="synthetic-parity-only",
    )
    snapshot, node_map = synthetic_shadow_snapshot(
        observable_scenario=observable_scenario,
        scenario_metadata=scenario_metadata,
        reference_window=reference_window,
        pseudonymizer=pseudonymizer,
    )
    actual = ServingFeatureBuilder().build_sequence([snapshot])[0]
    expected = normalized_reference_window(
        reference_window=reference_window,
        node_map=node_map,
    )
    actual_hash = canonical_hash(actual)
    expected_hash = canonical_hash(expected)
    differences = []
    for name in sorted(set(actual) | set(expected)):
        if actual.get(name) == expected.get(name):
            continue
        if name == "node_features":
            for row_index, (actual_row, expected_row) in enumerate(
                zip(
                    actual.get(name, []),
                    expected.get(name, []),
                    strict=False,
                )
            ):
                for feature_index, (actual_value, expected_value) in enumerate(
                    zip(actual_row, expected_row, strict=False)
                ):
                    if actual_value != expected_value:
                        differences.append(
                            {
                                "field": name,
                                "row": row_index,
                                "feature_index": feature_index,
                                "feature_name": actual[
                                    "node_feature_names"
                                ][feature_index],
                                "actual": actual_value,
                                "expected": expected_value,
                            }
                        )
                        if len(differences) >= 10:
                            break
                if len(differences) >= 10:
                    break
        else:
            differences.append(
                {
                    "field": name,
                    "actual": actual.get(name),
                    "expected": expected.get(name),
                }
            )
        if len(differences) >= 10:
            break
    return {
        "matches": actual_hash == expected_hash,
        "actual_feature_hash": actual_hash,
        "expected_feature_hash": expected_hash,
        "snapshot_hash": snapshot.content_hash(),
        "window_id": reference_window["window_id"],
        "first_differences": differences,
    }
