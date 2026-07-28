from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import reduce
from operator import mul
from statistics import mean, pstdev

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.nrim.simulation.feature_schema import (
    NODE_TYPES,
    SIGNAL_APPLICABLE_NODE_TYPES,
    SIGNAL_NAMES,
    build_feature_schema,
)

from ..hashing import canonical_hash
from .batch_adapter import BatchInputRecord
from .contracts import utc
from .signal_registry import (
    AvailabilityClass,
    SignalRegistry,
    SupportConsequence,
)
from .topology import TopologySnapshot


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ReconstructionState(str, Enum):
    COMPLETE = "COMPLETE"
    UNKNOWN = "UNKNOWN"
    ESCALATE = "ESCALATE"
    BLOCKED = "BLOCKED"


class FeatureEvidenceState(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


class FeatureLineage(StrictModel):
    feature_name: str
    state: FeatureEvidenceState
    source_ids: tuple[str, ...] = ()
    source_record_sha256: tuple[str, ...] = ()
    event_cutoff_utc: datetime
    knowledge_cutoff_utc: datetime
    aggregation: str
    applicability_evidence: str


class NodeFeatureLineage(StrictModel):
    topology_node_id: str
    observation_component_pseudonym: str | None
    feature_name: str
    state: FeatureEvidenceState
    source_ids: tuple[str, ...] = ()
    source_record_sha256: tuple[str, ...] = ()
    event_cutoff_utc: datetime
    knowledge_cutoff_utc: datetime
    topology_snapshot_sha256: str
    topology_capture_ids: tuple[str, ...]
    required_for_gate: bool
    applicability_evidence: str
    temporal_evidence: str


class TensorEvidence(StrictModel):
    tensor_name: str
    feature_names: tuple[str, ...]
    shape: tuple[int, ...] = Field(min_length=1)
    values: tuple[float | None, ...]
    observed_mask: tuple[int, ...]

    @model_validator(mode="after")
    def validate_tensor(self):
        size = reduce(mul, self.shape, 1)
        if len(self.values) != size or len(self.observed_mask) != size:
            raise ValueError("Tensor shape and flattened values differ")
        if len(self.feature_names) not in {self.shape[-1], size}:
            raise ValueError("Tensor feature-name axis differs")
        if any(item not in {0, 1} for item in self.observed_mask):
            raise ValueError("Observed mask must be binary")
        if any(
            value is not None and not math.isfinite(value)
            for value in self.values
        ):
            raise ValueError("Tensor evidence values must be finite")
        if any(
            (value is None) == bool(mask)
            for value, mask in zip(
                self.values,
                self.observed_mask,
                strict=True,
            )
        ):
            raise ValueError("Tensor values and masks are inconsistent")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class FeatureDryRunResult(StrictModel):
    state: ReconstructionState
    reason_codes: tuple[str, ...]
    frozen_feature_schema_sha256: str
    stage1: TensorEvidence
    residual: TensorEvidence
    stage2: TensorEvidence
    lineage: tuple[FeatureLineage, ...]
    per_node_lineage: tuple[NodeFeatureLineage, ...]
    inference_call_sha256: tuple[str, ...] = ()
    tuning_event_sha256: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_node_evidence(self):
        expected = self.stage2.shape[0] * self.stage2.shape[1]
        if len(self.per_node_lineage) != expected:
            raise ValueError("Stage 2 lineage does not cover every tensor cell")
        if self.stage2.shape[1] != len(self.stage2.feature_names):
            raise ValueError("Stage 2 feature axis differs from its schema")
        node_groups: list[str] = []
        for index, (value, lineage) in enumerate(
            zip(
                self.stage2.values,
                self.per_node_lineage,
                strict=True,
            )
        ):
            expected_feature = self.stage2.feature_names[
                index % self.stage2.shape[1]
            ]
            if lineage.feature_name != expected_feature:
                raise ValueError("Stage 2 lineage feature order differs")
            if (value is not None) != (
                lineage.state is FeatureEvidenceState.OBSERVED
            ):
                raise ValueError("Stage 2 values and lineage states differ")
            if lineage.required_for_gate and lineage.state is (
                FeatureEvidenceState.NOT_APPLICABLE
            ):
                raise ValueError(
                    "Required Stage 2 evidence cannot be not applicable"
                )
            if index % self.stage2.shape[1] == 0:
                node_groups.append(lineage.topology_node_id)
            elif lineage.topology_node_id != node_groups[-1]:
                raise ValueError("Stage 2 node lineage is not contiguous")
        if len(node_groups) != self.stage2.shape[0] or len(
            set(node_groups)
        ) != len(node_groups):
            raise ValueError("Stage 2 node axis differs from its lineage")
        if self.per_node_lineage:
            first = self.per_node_lineage[0]
            for lineage in self.per_node_lineage[1:]:
                if (
                    lineage.event_cutoff_utc != first.event_cutoff_utc
                    or lineage.knowledge_cutoff_utc
                    != first.knowledge_cutoff_utc
                    or lineage.topology_snapshot_sha256
                    != first.topology_snapshot_sha256
                    or lineage.topology_capture_ids
                    != first.topology_capture_ids
                ):
                    raise ValueError(
                        "Stage 2 lineage uses inconsistent reconstruction evidence"
                    )
        for digest in (
            *self.inference_call_sha256,
            *self.tuning_event_sha256,
        ):
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("Execution evidence commitment is invalid")
        return self

    @property
    def required_feature_fraction(self) -> float:
        required = tuple(
            item for item in self.per_node_lineage if item.required_for_gate
        )
        if not required:
            return 0.0
        return sum(
            item.state is FeatureEvidenceState.OBSERVED for item in required
        ) / len(required)

    @property
    def inference_call_count(self) -> int:
        return len(self.inference_call_sha256)

    @property
    def tuning_event_count(self) -> int:
        return len(self.tuning_event_sha256)

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class V06Stage2Node:
    node_id: str
    node_type: str

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError("Unknown frozen v0.6 node type")


@dataclass(frozen=True)
class V06Stage2ContextEvent:
    component_node_id: str | None
    event_time_utc: datetime
    ingestion_time_utc: datetime


_V06_CRITICAL_NODE_TYPES = {
    "middleware",
    "database",
    "catchup_service",
    "catchup_storage",
    "core_switch",
}
_V06_LOW_QUALITY_VALUES = {"low", "quarantined"}


def _v06_linear_slope(
    timestamps: list[datetime],
    values: list[float],
) -> float:
    if len(values) < 2:
        return 0.0
    start = timestamps[0]
    x_values = [
        (timestamp - start).total_seconds() / 3600.0
        for timestamp in timestamps
    ]
    x_mean = mean(x_values)
    y_mean = mean(values)
    denominator = sum((item - x_mean) ** 2 for item in x_values)
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, values, strict=True)
    )
    return numerator / denominator


def _v06_signal_statistics(
    *,
    records: tuple[BatchInputRecord, ...],
    cutoff: datetime,
    latest: float,
) -> dict[str, float]:
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.event_time_utc,
                item.ingestion_time_utc,
                item.source_id,
                item.source_sequence_id,
            ),
        )
    )
    values = [float(item.metric_value) for item in ordered]
    timestamps = [item.event_time_utc for item in ordered]
    low_quality_count = sum(
        item.quality in _V06_LOW_QUALITY_VALUES for item in ordered
    )
    return {
        "latest": latest,
        "mean": mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": (
            pstdev(values) if len(values) > 1 else 0.0
        ),
        "slope": _v06_linear_slope(timestamps, values),
        "count": float(len(values)),
        "low_quality_fraction": low_quality_count / len(values),
        "hours_since_latest": max(
            0.0,
            (cutoff - timestamps[-1]).total_seconds() / 3600.0,
        ),
    }


def reconstruct_v06_stage2_tensor(
    *,
    records: tuple[BatchInputRecord, ...],
    registry: SignalRegistry,
    topology: TopologySnapshot,
    nodes: tuple[V06Stage2Node, ...],
    context_events: tuple[V06Stage2ContextEvent, ...],
    observation_start_utc: datetime,
    event_cutoff_utc: datetime,
    knowledge_cutoff_utc: datetime,
) -> TensorEvidence:
    """Build the frozen 114-column Stage 2 tensor through IPTV-P0 evidence.

    This compatibility reconstruction is intentionally independent of the
    legacy serving builder. It first runs the governed IPTV-P0 local feature
    reconstruction, then expands those component-local states and values into
    the frozen v0.6 Stage 2 feature contract.
    """

    start = utc(observation_start_utc)
    event_cutoff = utc(event_cutoff_utc)
    knowledge_cutoff = utc(knowledge_cutoff_utc)
    if start > event_cutoff or knowledge_cutoff < event_cutoff:
        raise ValueError("Stage 2 bitemporal window is invalid")
    topology_node_ids = {item.node_id for item in topology.nodes}
    candidate_ids = tuple(item.node_id for item in nodes)
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or set(candidate_ids) != topology_node_ids
    ):
        raise ValueError(
            "Stage 2 candidate order differs from reconstructed topology"
        )
    visible_records = tuple(
        record
        for record in records
        if start <= record.event_time_utc <= event_cutoff
        and record.ingestion_time_utc <= knowledge_cutoff
    )
    local_result = reconstruct_frozen_features(
        records=visible_records,
        registry=registry,
        topology=topology,
        event_cutoff_utc=event_cutoff,
        knowledge_cutoff_utc=knowledge_cutoff,
    )
    if local_result.stage2.feature_names != tuple(SIGNAL_NAMES):
        raise ValueError("IPTV-P0 local feature order differs")
    if local_result.stage2.shape != (
        len(topology.nodes),
        len(SIGNAL_NAMES),
    ):
        raise ValueError("IPTV-P0 local Stage 2 shape differs")
    local_cells = {
        (lineage.topology_node_id, lineage.feature_name): (value, lineage)
        for value, lineage in zip(
            local_result.stage2.values,
            local_result.per_node_lineage,
            strict=True,
        )
    }
    if len(local_cells) != len(topology.nodes) * len(SIGNAL_NAMES):
        raise ValueError("IPTV-P0 local Stage 2 lineage is incomplete")

    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    for edge in topology.edges:
        outgoing[edge.source_node_id] += 1
        incoming[edge.destination_node_id] += 1
    visible_context = tuple(
        event
        for event in context_events
        if start <= utc(event.event_time_utc) <= event_cutoff
        and utc(event.ingestion_time_utc) <= knowledge_cutoff
    )
    global_change_count = len(visible_context)
    changes_by_node: dict[str, int] = defaultdict(int)
    for event in visible_context:
        if event.component_node_id is not None:
            changes_by_node[event.component_node_id] += 1

    records_by_node_signal: dict[
        tuple[str, str],
        list[BatchInputRecord],
    ] = defaultdict(list)
    component_by_node = {
        item.node_id: item.observation_component_pseudonym
        for item in topology.nodes
    }
    node_by_component = {
        component: node_id
        for node_id, component in component_by_node.items()
        if component is not None
    }
    for record in visible_records:
        if record.applicability != "observed":
            continue
        node_id = node_by_component.get(record.component_pseudonym)
        if node_id is not None:
            records_by_node_signal[(node_id, record.metric_name)].append(
                record
            )

    schema = build_feature_schema()
    feature_names = tuple(
        definition.name for definition in schema.node_features
    )
    defaults = {
        definition.name: definition.default_value
        for definition in schema.node_features
    }
    values: list[float] = []
    for node in nodes:
        row = dict(defaults)
        for known_type in NODE_TYPES:
            row[f"node_type__{known_type}"] = float(
                node.node_type == known_type
            )
        row["in_degree"] = float(incoming[node.node_id])
        row["out_degree"] = float(outgoing[node.node_id])
        row["total_degree"] = float(
            incoming[node.node_id] + outgoing[node.node_id]
        )
        row["critical_service"] = float(
            node.node_type in _V06_CRITICAL_NODE_TYPES
        )
        row["recent_change_event_count"] = float(
            changes_by_node[node.node_id] + global_change_count
        )
        for signal_name in SIGNAL_NAMES:
            safe_signal = signal_name.replace(".", "__")
            local_value, lineage = local_cells[(node.node_id, signal_name)]
            applicable = (
                node.node_type
                in SIGNAL_APPLICABLE_NODE_TYPES[signal_name]
            )
            if applicable != (
                lineage.state is not FeatureEvidenceState.NOT_APPLICABLE
            ):
                raise ValueError(
                    "IPTV-P0 applicability differs from frozen v0.6 topology"
                )
            row[f"{safe_signal}__applicable"] = float(applicable)
            if not applicable:
                row[f"{safe_signal}__missing"] = 0.0
                continue
            if lineage.state is not FeatureEvidenceState.OBSERVED:
                row[f"{safe_signal}__missing"] = 1.0
                continue
            if local_value is None:
                raise ValueError("Observed IPTV-P0 feature has no value")
            local_records = tuple(
                records_by_node_signal[(node.node_id, signal_name)]
            )
            if not local_records:
                raise ValueError("Observed IPTV-P0 feature has no lineage")
            statistics = _v06_signal_statistics(
                records=local_records,
                cutoff=event_cutoff,
                latest=float(local_value),
            )
            for statistic, value in statistics.items():
                row[f"{safe_signal}__{statistic}"] = value
            row[f"{safe_signal}__missing"] = 0.0
        values.extend(float(row[name]) for name in feature_names)

    return TensorEvidence(
        tensor_name="v06_stage2_input",
        feature_names=feature_names,
        shape=(len(nodes), len(feature_names)),
        values=tuple(values),
        observed_mask=(1,) * len(values),
    )


def reconstruct_frozen_features(
    *,
    records: tuple[BatchInputRecord, ...],
    registry: SignalRegistry,
    topology: TopologySnapshot,
    event_cutoff_utc: datetime,
    knowledge_cutoff_utc: datetime,
    conditional_predicates: dict[str, bool | None] | None = None,
) -> FeatureDryRunResult:
    event_cutoff = utc(event_cutoff_utc)
    knowledge_cutoff = utc(knowledge_cutoff_utc)
    conditional_predicates = conditional_predicates or {}
    visible = tuple(
        record
        for record in records
        if record.event_time_utc <= event_cutoff
        and record.ingestion_time_utc <= knowledge_cutoff
    )
    reasons: list[str] = []
    lineage: list[FeatureLineage] = []
    stage1_values: list[float | None] = []
    residual_values: list[float | None] = []
    for signal_name in SIGNAL_NAMES:
        requirement = registry.requirement(signal_name)
        predicate = conditional_predicates.get(signal_name)
        applies = not (
            requirement.availability is AvailabilityClass.CONDITIONAL
            and predicate is False
        )
        signal_records = [
            record
            for record in visible
            if record.metric_name == signal_name
            and record.applicability == "observed"
        ]
        signal_records.sort(
            key=lambda item: (
                item.event_time_utc,
                item.ingestion_time_utc,
                item.source_id,
                item.source_sequence_id,
            )
        )
        latest_by_component: dict[str, BatchInputRecord] = {}
        prior_by_component: dict[str, BatchInputRecord] = {}
        for record in signal_records:
            previous = latest_by_component.get(record.component_pseudonym)
            if previous is not None:
                prior_by_component[record.component_pseudonym] = previous
            latest_by_component[record.component_pseudonym] = record

        if requirement.availability is AvailabilityClass.UNAVAILABLE:
            value = None
            residual = None
            state = FeatureEvidenceState.UNAVAILABLE
            evidence = "deployment_registry_declares_unavailable"
            route = requirement.support_consequence
            reasons.append(f"{route.value}:{signal_name}:UNAVAILABLE")
        elif not applies:
            value = None
            residual = None
            state = FeatureEvidenceState.NOT_APPLICABLE
            evidence = "registered_conditional_predicate_false"
        elif latest_by_component:
            current = [
                float(record.metric_value)
                for record in latest_by_component.values()
                if record.metric_value is not None
            ]
            previous = [
                float(record.metric_value)
                for record in prior_by_component.values()
                if record.metric_value is not None
            ]
            value = _mean(current)
            previous_value = _mean(previous)
            residual = (
                None
                if value is None or previous_value is None
                else value - previous_value
            )
            state = FeatureEvidenceState.OBSERVED
            evidence = "qualified_records_visible_at_bitemporal_cutoffs"
        else:
            value = None
            residual = None
            state = FeatureEvidenceState.MISSING
            evidence = "no_qualified_record_visible_at_cutoffs"
            route = registry.route_absence(
                signal_name,
                predicate_value=predicate,
            )
            if route is not None:
                reasons.append(f"{route.value}:{signal_name}:MISSING")
        stage1_values.append(value)
        residual_values.append(residual)
        source_records = tuple(
            latest_by_component[key]
            for key in sorted(latest_by_component)
        )
        lineage.append(
            FeatureLineage(
                feature_name=signal_name,
                state=state,
                source_ids=tuple(
                    sorted({record.source_id for record in source_records})
                ),
                source_record_sha256=tuple(
                    record.record_sha256 for record in source_records
                ),
                event_cutoff_utc=event_cutoff,
                knowledge_cutoff_utc=knowledge_cutoff,
                aggregation=requirement.aggregation,
                applicability_evidence=evidence,
            )
        )

    stage1 = TensorEvidence(
        tensor_name="stage1_input",
        feature_names=tuple(SIGNAL_NAMES),
        shape=(len(SIGNAL_NAMES),),
        values=tuple(stage1_values),
        observed_mask=tuple(int(value is not None) for value in stage1_values),
    )
    residual = TensorEvidence(
        tensor_name="healthy_residual_input",
        feature_names=tuple(SIGNAL_NAMES),
        shape=(len(SIGNAL_NAMES),),
        values=tuple(residual_values),
        observed_mask=tuple(
            int(value is not None) for value in residual_values
        ),
    )
    node_ids = tuple(item.node_id for item in topology.nodes)
    stage2_values: list[float | None] = []
    per_node_lineage: list[NodeFeatureLineage] = []
    topology_sha256 = topology.content_hash
    for node in topology.nodes:
        for signal_name in SIGNAL_NAMES:
            requirement = registry.requirement(signal_name)
            predicate = conditional_predicates.get(signal_name)
            registered_applicable = signal_name in node.applicable_signals
            condition_applies = not (
                requirement.availability
                is AvailabilityClass.CONDITIONAL
                and predicate is False
            )
            local_records = [
                record
                for record in visible
                if record.metric_name == signal_name
                and record.component_pseudonym
                == node.observation_component_pseudonym
                and record.applicability == "observed"
            ]
            local_records.sort(
                key=lambda item: (
                    item.event_time_utc,
                    item.ingestion_time_utc,
                    item.source_id,
                    item.source_sequence_id,
                )
            )
            if not registered_applicable or not condition_applies:
                value = None
                state = FeatureEvidenceState.NOT_APPLICABLE
                applicability_evidence = (
                    "topology_node_signal_not_applicable"
                    if not registered_applicable
                    else "registered_conditional_predicate_false"
                )
                selected: tuple[BatchInputRecord, ...] = ()
            elif requirement.availability is AvailabilityClass.UNAVAILABLE:
                value = None
                state = FeatureEvidenceState.UNAVAILABLE
                applicability_evidence = (
                    "deployment_registry_declares_unavailable"
                )
                selected = ()
                reasons.append(
                    f"{requirement.support_consequence.value}:"
                    f"{node.node_id}:{signal_name}:UNAVAILABLE"
                )
            elif local_records:
                latest = local_records[-1]
                value = (
                    None
                    if latest.metric_value is None
                    else float(latest.metric_value)
                )
                state = FeatureEvidenceState.OBSERVED
                applicability_evidence = (
                    "topology_binding_and_signal_applicability_registered"
                )
                selected = (latest,)
            else:
                value = None
                state = FeatureEvidenceState.MISSING
                applicability_evidence = (
                    "applicable_topology_node_has_no_visible_local_record"
                )
                selected = ()
                route = registry.route_absence(
                    signal_name,
                    predicate_value=predicate,
                )
                if route is not None:
                    reasons.append(
                        f"{route.value}:{node.node_id}:"
                        f"{signal_name}:MISSING"
                    )
            required_for_gate = (
                registered_applicable
                and condition_applies
                and (
                    requirement.availability
                    in {
                        AvailabilityClass.REQUIRED,
                        AvailabilityClass.CONDITIONAL,
                    }
                )
            )
            stage2_values.append(value)
            per_node_lineage.append(
                NodeFeatureLineage(
                    topology_node_id=node.node_id,
                    observation_component_pseudonym=(
                        node.observation_component_pseudonym
                    ),
                    feature_name=signal_name,
                    state=state,
                    source_ids=tuple(
                        sorted({record.source_id for record in selected})
                    ),
                    source_record_sha256=tuple(
                        record.record_sha256 for record in selected
                    ),
                    event_cutoff_utc=event_cutoff,
                    knowledge_cutoff_utc=knowledge_cutoff,
                    topology_snapshot_sha256=topology_sha256,
                    topology_capture_ids=topology.applied_capture_ids,
                    required_for_gate=required_for_gate,
                    applicability_evidence=applicability_evidence,
                    temporal_evidence=(
                        "event_time<=event_cutoff_and_"
                        "ingestion_time<=knowledge_cutoff"
                    ),
                )
            )
    stage2 = TensorEvidence(
        tensor_name="stage2_input",
        feature_names=tuple(SIGNAL_NAMES),
        shape=(len(node_ids), len(SIGNAL_NAMES)),
        values=tuple(stage2_values),
        observed_mask=tuple(
            int(value is not None) for value in stage2_values
        ),
    )
    if any(reason.startswith("BLOCKED:") for reason in reasons):
        state = ReconstructionState.BLOCKED
    elif any(reason.startswith("ESCALATE:") for reason in reasons):
        state = ReconstructionState.ESCALATE
    elif any(reason.startswith("UNKNOWN:") for reason in reasons):
        state = ReconstructionState.UNKNOWN
    else:
        state = ReconstructionState.COMPLETE
    return FeatureDryRunResult(
        state=state,
        reason_codes=tuple(sorted(set(reasons))),
        frozen_feature_schema_sha256=canonical_hash(
            build_feature_schema().model_dump(mode="json")
        ),
        stage1=stage1,
        residual=residual,
        stage2=stage2,
        lineage=tuple(lineage),
        per_node_lineage=tuple(per_node_lineage),
    )
