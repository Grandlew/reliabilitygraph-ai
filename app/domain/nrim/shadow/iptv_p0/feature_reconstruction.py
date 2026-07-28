from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from functools import reduce
from operator import mul

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.nrim.simulation.feature_schema import (
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
