from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from app.domain.nrim.simulation.feature_schema import (
    SIGNAL_APPLICABLE_NODE_TYPES,
    SIGNAL_NAMES,
    FeatureSchema,
    build_feature_schema,
)
from app.domain.nrim.simulation.graph_feature_builder import (
    build_edge_feature_matrix,
    build_node_feature_matrix,
)
from app.domain.nrim.simulation.model_ready_exporter import (
    add_causal_history_features,
)

from .contracts import (
    Applicability,
    DataQualityState,
    DeploymentProfile,
    MetricName,
    OperationalEvent,
    ServingSnapshot,
    SnapshotMode,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from .hashing import canonical_hash
from .privacy import assert_inference_payload_is_label_free
from .store import AppendOnlyEvidenceStore


@dataclass(frozen=True)
class WatermarkPolicy:
    """Registered per-family event-time lateness and freshness contract."""

    default_allowed_lateness: timedelta = timedelta(minutes=5)
    allowed_lateness_by_metric: dict[MetricName, timedelta] = field(
        default_factory=dict
    )
    default_max_freshness: timedelta = timedelta(hours=2)
    max_freshness_by_metric: dict[MetricName, timedelta] = field(
        default_factory=dict
    )
    minimum_available_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.default_allowed_lateness.total_seconds() < 0:
            raise ValueError("Allowed lateness cannot be negative")
        if self.default_max_freshness.total_seconds() <= 0:
            raise ValueError("Maximum freshness must be positive")
        if not 0.0 <= self.minimum_available_fraction <= 1.0:
            raise ValueError("Availability fraction must be in [0, 1]")
        if any(
            value.total_seconds() < 0
            for value in self.allowed_lateness_by_metric.values()
        ):
            raise ValueError("Allowed lateness cannot be negative")
        if any(
            value.total_seconds() <= 0
            for value in self.max_freshness_by_metric.values()
        ):
            raise ValueError("Maximum freshness must be positive")

    def allowed_lateness(self, metric: MetricName) -> timedelta:
        return self.allowed_lateness_by_metric.get(
            metric,
            self.default_allowed_lateness,
        )

    def max_freshness(self, metric: MetricName) -> timedelta:
        return self.max_freshness_by_metric.get(
            metric,
            self.default_max_freshness,
        )

    @property
    def maximum_lateness(self) -> timedelta:
        return max(
            (
                self.default_allowed_lateness,
                *self.allowed_lateness_by_metric.values(),
            )
        )


class TopologyRegistry:
    """Versioned topology/profile registry with overlap rejection."""

    def __init__(
        self,
        *,
        components: Iterable[TopologyComponent],
        edges: Iterable[TopologyEdge],
        profiles: Iterable[DeploymentProfile],
    ) -> None:
        self.components = tuple(components)
        self.edges = tuple(edges)
        self.profiles = tuple(profiles)
        self._validate_no_component_overlap()
        self._validate_no_profile_overlap()

    @classmethod
    def from_store(
        cls,
        store: AppendOnlyEvidenceStore,
        *,
        deployment_pseudonym: str | None = None,
    ) -> "TopologyRegistry":
        components, edges, profiles = store.topology_contracts(
            deployment_pseudonym=deployment_pseudonym
        )
        return cls(
            components=components,
            edges=edges,
            profiles=profiles,
        )

    @staticmethod
    def _overlap(
        left_start: datetime,
        left_end: datetime | None,
        right_start: datetime,
        right_end: datetime | None,
    ) -> bool:
        maximum = datetime.max.replace(tzinfo=left_start.tzinfo)
        return left_start < (right_end or maximum) and right_start < (
            left_end or maximum
        )

    def _validate_no_component_overlap(self) -> None:
        grouped: dict[tuple[str, str], list[TopologyComponent]] = defaultdict(
            list
        )
        for component in self.components:
            grouped[
                (
                    component.deployment_pseudonym,
                    component.component_pseudonym,
                )
            ].append(component)
        for key, rows in grouped.items():
            ordered = sorted(rows, key=lambda item: item.valid_from_utc)
            for left, right in zip(ordered, ordered[1:], strict=False):
                if self._overlap(
                    left.valid_from_utc,
                    left.valid_to_utc,
                    right.valid_from_utc,
                    right.valid_to_utc,
                ):
                    raise ValueError(
                        f"Overlapping component validity intervals: {key}"
                    )

    def _validate_no_profile_overlap(self) -> None:
        grouped: dict[str, list[DeploymentProfile]] = defaultdict(list)
        for profile in self.profiles:
            grouped[profile.deployment_pseudonym].append(profile)
        for deployment, rows in grouped.items():
            ordered = sorted(rows, key=lambda item: item.valid_from_utc)
            for left, right in zip(ordered, ordered[1:], strict=False):
                if self._overlap(
                    left.valid_from_utc,
                    left.valid_to_utc,
                    right.valid_from_utc,
                    right.valid_to_utc,
                ):
                    raise ValueError(
                        "Overlapping profile validity intervals: "
                        + deployment
                    )

    @staticmethod
    def _active(
        valid_from: datetime,
        valid_to: datetime | None,
        cutoff: datetime,
    ) -> bool:
        return valid_from <= cutoff and (
            valid_to is None or cutoff < valid_to
        )

    def snapshot(
        self,
        *,
        deployment_pseudonym: str,
        cutoff_utc: datetime,
    ) -> tuple[
        str,
        DeploymentProfile,
        tuple[TopologyComponent, ...],
        tuple[TopologyEdge, ...],
    ]:
        components = tuple(
            item
            for item in self.components
            if item.deployment_pseudonym == deployment_pseudonym
            and self._active(
                item.valid_from_utc,
                item.valid_to_utc,
                cutoff_utc,
            )
        )
        profiles = tuple(
            item
            for item in self.profiles
            if item.deployment_pseudonym == deployment_pseudonym
            and self._active(
                item.valid_from_utc,
                item.valid_to_utc,
                cutoff_utc,
            )
        )
        if not components:
            raise ValueError("No active topology at decision cutoff")
        if len(profiles) != 1:
            raise ValueError("Exactly one active deployment profile is required")
        versions = {item.topology_version for item in components}
        if len(versions) != 1:
            raise ValueError("Multiple active topology versions at cutoff")
        version = next(iter(versions))
        edges = tuple(
            item
            for item in self.edges
            if item.deployment_pseudonym == deployment_pseudonym
            and item.topology_version == version
            and self._active(
                item.valid_from_utc,
                item.valid_to_utc,
                cutoff_utc,
            )
        )
        component_ids = {
            item.component_pseudonym for item in components
        }
        if any(
            edge.source_component not in component_ids
            or edge.destination_component not in component_ids
            for edge in edges
        ):
            raise ValueError("Active edge references inactive component")
        return version, profiles[0], components, edges


class EventTimeReplayEngine:
    def __init__(
        self,
        *,
        store: AppendOnlyEvidenceStore,
        topology_registry: TopologyRegistry,
        watermark_policy: WatermarkPolicy,
    ) -> None:
        self.store = store
        self.topology_registry = topology_registry
        self.watermark_policy = watermark_policy

    def build_snapshot(
        self,
        *,
        deployment_pseudonym: str,
        observation_start_utc: datetime,
        decision_cutoff_utc: datetime,
        as_of_ingestion_time_utc: datetime | None = None,
        mode: SnapshotMode = SnapshotMode.PROSPECTIVE,
        supersedes_prediction_id: str | None = None,
    ) -> ServingSnapshot:
        if as_of_ingestion_time_utc is None:
            as_of_ingestion_time_utc = (
                decision_cutoff_utc
                + self.watermark_policy.maximum_lateness
            )
        (
            topology_version,
            profile,
            components,
            edges,
        ) = self.topology_registry.snapshot(
            deployment_pseudonym=deployment_pseudonym,
            cutoff_utc=decision_cutoff_utc,
        )
        observations = self.store.telemetry_between(
            deployment_pseudonym=deployment_pseudonym,
            event_start_utc=observation_start_utc,
            event_end_utc=decision_cutoff_utc,
            ingested_by_utc=as_of_ingestion_time_utc,
        )
        events = self.store.operational_events_between(
            deployment_pseudonym=deployment_pseudonym,
            event_start_utc=observation_start_utc,
            event_end_utc=decision_cutoff_utc,
            ingested_by_utc=as_of_ingestion_time_utc,
        )
        if mode is SnapshotMode.PROSPECTIVE:
            observations = [
                item
                for item in observations
                if item.ingestion_time_utc
                <= decision_cutoff_utc
                + self.watermark_policy.allowed_lateness(item.metric_name)
            ]
            events = [
                item
                for item in events
                if item.ingestion_time_utc
                <= decision_cutoff_utc
                + self.watermark_policy.default_allowed_lateness
            ]

        component_by_id = {
            item.component_pseudonym: item
            for item in components
        }
        expected_pairs = {
            (component.component_pseudonym, MetricName(signal))
            for component in components
            for signal in SIGNAL_NAMES
            if component.component_type.value
            in SIGNAL_APPLICABLE_NODE_TYPES[signal]
        }
        latest_observed: dict[
            tuple[str, MetricName],
            TelemetryObservation,
        ] = {}
        warnings: list[str] = []
        for item in observations:
            if item.topology_version != topology_version:
                warnings.append(
                    "topology_version_mismatch:"
                    + item.source_sequence_id
                )
                continue
            component = component_by_id.get(item.component_pseudonym)
            if component is None:
                warnings.append(
                    "unknown_component:" + item.source_sequence_id
                )
                continue
            if component.component_type is not item.component_type:
                warnings.append(
                    "component_type_mismatch:"
                    + item.source_sequence_id
                )
                continue
            if item.applicability is Applicability.OBSERVED:
                key = (item.component_pseudonym, item.metric_name)
                if (
                    key not in latest_observed
                    or latest_observed[key].event_time_utc
                    < item.event_time_utc
                ):
                    latest_observed[key] = item
        available_pairs = set(latest_observed)
        missing_pairs = expected_pairs - available_pairs
        for component, metric in sorted(
            missing_pairs,
            key=lambda item: (item[0], item[1].value),
        ):
            warnings.append(f"missing:{component}:{metric.value}")
        stale_pairs = []
        for key, item in latest_observed.items():
            if (
                decision_cutoff_utc - item.event_time_utc
                > self.watermark_policy.max_freshness(item.metric_name)
            ):
                stale_pairs.append(key)
                warnings.append(
                    f"stale:{key[0]}:{key[1].value}"
                )
        available_count = len(available_pairs - set(stale_pairs))
        expected_count = len(expected_pairs)
        available_fraction = (
            available_count / expected_count if expected_count else 0.0
        )
        blocking = [
            item
            for item in warnings
            if item.startswith(
                (
                    "topology_version_mismatch:",
                    "unknown_component:",
                    "component_type_mismatch:",
                )
            )
        ]
        if available_fraction < self.watermark_policy.minimum_available_fraction:
            blocking.append(
                "available_fraction_below_minimum:"
                f"{available_fraction:.6f}"
            )
        if blocking:
            quality_state = DataQualityState.BLOCKED
        elif warnings:
            quality_state = DataQualityState.DEGRADED
        else:
            quality_state = DataQualityState.ACCEPTED
        identity = {
            "deployment": deployment_pseudonym,
            "start": observation_start_utc,
            "cutoff": decision_cutoff_utc,
            "as_of": as_of_ingestion_time_utc,
            "mode": mode,
            "topology_version": topology_version,
            "observation_ids": [
                (item.collector_id, item.source_sequence_id)
                for item in observations
            ],
        }
        return ServingSnapshot(
            snapshot_id="snapshot_" + canonical_hash(identity)[:32],
            deployment_pseudonym=deployment_pseudonym,
            observation_start_utc=observation_start_utc,
            decision_cutoff_utc=decision_cutoff_utc,
            as_of_ingestion_time_utc=as_of_ingestion_time_utc,
            topology_version=topology_version,
            profile=profile,
            components=components,
            edges=edges,
            observations=tuple(observations),
            operational_events=tuple(events),
            data_quality_state=quality_state,
            data_quality_warnings=tuple(sorted(set(warnings + blocking))),
            expected_observation_count=expected_count,
            available_observation_count=available_count,
            mode=mode,
            supersedes_prediction_id=supersedes_prediction_id,
        )


@dataclass
class FeatureHistoryState:
    values: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    persistence: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )


class ServingFeatureBuilder:
    """Single deterministic builder used by online and historical replay."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or build_feature_schema()

    @property
    def schema_hash(self) -> str:
        return canonical_hash(self.schema.model_dump(mode="json"))

    @staticmethod
    def _topology(snapshot: ServingSnapshot) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": item.component_pseudonym,
                    "node_type": item.component_type.value,
                }
                for item in snapshot.components
            ],
            "edges": [
                {
                    "source_node_id": item.source_component,
                    "target_node_id": item.destination_component,
                    "edge_type": item.dependency_type.value,
                    "propagation_delay_minutes": (
                        item.propagation_delay_minutes
                    ),
                    "propagation_strength": item.propagation_strength,
                }
                for item in snapshot.edges
            ],
        }

    @staticmethod
    def _telemetry(snapshot: ServingSnapshot) -> list[dict[str, Any]]:
        return [
            {
                "component_node_id": item.component_pseudonym,
                "signal_name": item.metric_name.value,
                "value": item.metric_value,
                "observed_at": item.event_time_utc.isoformat(),
                "quality": item.quality.value,
            }
            for item in snapshot.observations
            if item.applicability is not Applicability.NOT_APPLICABLE
        ]

    @staticmethod
    def _events(snapshot: ServingSnapshot) -> list[dict[str, Any]]:
        return [
            {
                "component_node_id": item.component_pseudonym,
                "signal_name": item.signal_name,
                "observed_at": item.event_time_utc.isoformat(),
            }
            for item in snapshot.operational_events
        ]

    def build(
        self,
        *,
        snapshot: ServingSnapshot,
        history: FeatureHistoryState,
    ) -> dict[str, Any]:
        if snapshot.data_quality_state is DataQualityState.BLOCKED:
            raise ValueError("Blocked snapshots cannot enter feature inference")
        topology = self._topology(snapshot)
        node_ids, node_features = build_node_feature_matrix(
            topology=topology,
            telemetry=self._telemetry(snapshot),
            context_events=self._events(snapshot),
            cutoff=snapshot.decision_cutoff_utc,
            schema=self.schema,
        )
        node_feature_names = [
            definition.name
            for definition in self.schema.node_features
        ]
        add_causal_history_features(
            node_ids=node_ids,
            feature_names=node_feature_names,
            node_features=node_features,
            history=history.values,
            persistence=history.persistence,
        )
        edge_index, edge_features = build_edge_feature_matrix(
            topology=topology,
            node_ids=node_ids,
            schema=self.schema,
        )
        window = {
            "window_id": snapshot.snapshot_id,
            "observation_start": (
                snapshot.observation_start_utc.isoformat()
            ),
            "observation_cutoff": (
                snapshot.decision_cutoff_utc.isoformat()
            ),
            "node_ids": node_ids,
            "node_feature_names": node_feature_names,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_feature_names": [
                definition.name
                for definition in self.schema.edge_features
            ],
            "edge_features": edge_features,
        }
        assert_inference_payload_is_label_free(window)
        return window

    def build_sequence(
        self,
        snapshots: Sequence[ServingSnapshot],
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            snapshots,
            key=lambda item: (
                item.decision_cutoff_utc,
                item.snapshot_id,
            ),
        )
        if len(
            {item.deployment_pseudonym for item in ordered}
        ) > 1:
            raise ValueError("Feature sequence must contain one deployment")
        if any(
            item.data_quality_state is DataQualityState.BLOCKED
            for item in ordered
        ):
            raise ValueError("Blocked snapshots cannot enter feature inference")
        history = FeatureHistoryState()
        return [
            self.build(snapshot=snapshot, history=history)
            for snapshot in ordered
        ]

    @staticmethod
    def feature_hash(window: dict[str, Any]) -> str:
        return canonical_hash(window)


def assert_online_replay_parity(
    *,
    online_window: dict[str, Any],
    replay_window: dict[str, Any],
) -> None:
    online_hash = ServingFeatureBuilder.feature_hash(online_window)
    replay_hash = ServingFeatureBuilder.feature_hash(replay_window)
    if online_hash != replay_hash:
        raise ValueError(
            "Online/replay feature parity failed: "
            f"{online_hash} != {replay_hash}"
        )
