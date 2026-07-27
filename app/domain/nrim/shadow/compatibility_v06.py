from __future__ import annotations

from pydantic import Field

from .candidate_evidence import (
    AlternativeHypothesis,
    CandidateExclusion,
    CandidateGeneratorIdentity,
    CandidateSet,
    CausalEvidenceSet,
    EvidenceApplicability,
    EvidenceObligation,
    EvidenceStatus,
    EvidenceTimeRelation,
    ExpectedDirection,
    StrictDecisionModel,
)
from .contracts import (
    DataQualityState,
    DecisionState,
    PredictionEnvelope,
    ServingSnapshot,
)
from .decision_envelope import seal_decision_stream
from .decision_events import (
    CandidateSetProducedPayload,
    CausalEvidenceProducedPayload,
    DataQualityAssessedPayload,
    DecisionEvent,
    DecisionEventType,
    DecisionSealedPayload,
    IncidentEvidenceProducedPayload,
    RankingProducedPayload,
    SnapshotSealedPayload,
    SupportAssessedPayload,
    SupportState,
    ZERO_SHA256,
)
from .hashing import canonical_hash, typed_hash


class CandidateContext(StrictDecisionModel):
    candidate_set: CandidateSet
    exclusion_provenance_complete: bool = False
    generator_notes: tuple[str, ...] = ()
    schema_version: str = "0.7.2"

    @classmethod
    def from_v06(
        cls,
        *,
        serving_snapshot: ServingSnapshot,
        prediction: PredictionEnvelope,
        exclusions: tuple[CandidateExclusion, ...] = (),
    ) -> CandidateContext:
        included = tuple(
            sorted(
                item.component_pseudonym
                for item in serving_snapshot.components
                if item.component_pseudonym
                not in {entry.component_pseudonym for entry in exclusions}
            )
        )
        generator = CandidateGeneratorIdentity(
            algorithm_version="v0.6-ranker-input-observer/1",
            topology_snapshot_sha256=prediction.topology_snapshot_hash,
            feature_schema_sha256=prediction.feature_schema_hash,
            policy_sha256=prediction.policy_hash,
            configuration_sha256=typed_hash(
                type_name="v06_candidate_configuration",
                schema_version="0.7.2",
                value={
                    "top_k": len(prediction.stage2_top_k),
                    "activation_path": prediction.activation_path,
                },
            ),
        )
        return cls(
            candidate_set=CandidateSet(
                decision_cutoff_utc=prediction.decision_cutoff_utc,
                included_candidates=included,
                exclusions=tuple(
                    sorted(
                        exclusions,
                        key=lambda item: item.component_pseudonym,
                    )
                ),
                generator=generator,
            ),
            exclusion_provenance_complete=False,
            generator_notes=("unavailable_v06_exclusions_are_unresolved",),
        )


def _support_state(prediction: PredictionEnvelope) -> SupportState:
    if prediction.final_decision in {
        DecisionState.UNKNOWN,
        DecisionState.ESCALATE,
    }:
        return SupportState.UNSUPPORTED
    if prediction.support_score is None:
        return SupportState.LIMITED
    return SupportState.IN_SUPPORT


def _evidence_set(
    prediction: PredictionEnvelope,
    *,
    incident_event_id: str,
) -> CausalEvidenceSet:
    ranked = tuple(
        item.component_pseudonym for item in prediction.stage2_top_k
    )
    obligations: list[EvidenceObligation] = []
    by_candidate: dict[str, list[str]] = {}
    for cause in prediction.stage2_top_k:
        evidence = cause.evidence or {"unresolved": 0.0}
        for mechanism, value in sorted(evidence.items()):
            status = (
                EvidenceStatus.SUPPORTS
                if value > 0.0
                else EvidenceStatus.CONTRADICTS
                if value < 0.0
                else EvidenceStatus.UNRESOLVED
            )
            source_ids = (
                (incident_event_id,)
                if status
                in {EvidenceStatus.SUPPORTS, EvidenceStatus.CONTRADICTS}
                else ()
            )
            obligation_id = "obligation_" + canonical_hash(
                {
                    "candidate": cause.component_pseudonym,
                    "mechanism": mechanism,
                    "value": value,
                    "source": incident_event_id,
                }
            )[:32]
            obligations.append(
                EvidenceObligation(
                    obligation_id=obligation_id,
                    candidate_pseudonym=cause.component_pseudonym,
                    mechanism_code=(
                        mechanism.lower()
                        .replace(" ", "_")
                        .replace("/", "_")
                    ),
                    expected_direction=ExpectedDirection.INCREASES,
                    time_relation=EvidenceTimeRelation.WINDOW_AGGREGATE,
                    applicability=EvidenceApplicability.APPLICABLE,
                    status=status,
                    source_event_ids=source_ids,
                    rationale_code="frozen_v06_observational_feature",
                    observed_value=float(value),
                    standardized_value=float(value),
                )
            )
            by_candidate.setdefault(
                cause.component_pseudonym,
                [],
            ).append(obligation_id)
    alternatives: list[AlternativeHypothesis] = []
    if len(ranked) > 1:
        for index, candidate in enumerate(ranked):
            alternative = ranked[(index + 1) % len(ranked)]
            alternatives.append(
                AlternativeHypothesis(
                    candidate_pseudonym=candidate,
                    alternative_candidate_pseudonym=alternative,
                    discriminator_obligation_ids=tuple(
                        by_candidate[candidate]
                    ),
                )
            )
    return CausalEvidenceSet(
        ranked_candidates=ranked,
        obligations=tuple(obligations),
        alternatives=tuple(alternatives),
        unresolved_reason_code=(
            "no_stage2_output" if not ranked else None
        ),
    )


def wrap_v06_prediction(
    *,
    serving_snapshot: ServingSnapshot,
    prediction: PredictionEnvelope,
    candidate_context: CandidateContext,
) -> tuple[DecisionEvent, ...]:
    if serving_snapshot.deployment_pseudonym != prediction.deployment_pseudonym:
        raise ValueError("Snapshot and prediction deployments differ")
    if serving_snapshot.decision_cutoff_utc != prediction.decision_cutoff_utc:
        raise ValueError("Snapshot and prediction cutoffs differ")
    if candidate_context.candidate_set.decision_cutoff_utc != (
        prediction.decision_cutoff_utc
    ):
        raise ValueError("Candidate context uses another decision cutoff")
    ranked = {
        item.component_pseudonym for item in prediction.stage2_top_k
    }
    if not ranked.issubset(
        candidate_context.candidate_set.included_candidates
    ):
        raise ValueError("Frozen rank contains a component outside CandidateSet")

    decision_id = "decision_" + canonical_hash(
        {
            "prediction_id": prediction.prediction_id,
            "prediction_sha256": canonical_hash(prediction),
        }
    )[:32]
    recorded = prediction.created_at_utc
    cutoff = prediction.decision_cutoff_utc
    correlation_id = prediction.prediction_id
    events: list[DecisionEvent] = []

    def append(
        event_type: DecisionEventType,
        payload,
        *,
        causation_event_id: str | None = None,
    ) -> DecisionEvent:
        previous = (
            events[-1].event_sha256 if events else ZERO_SHA256
        )
        event = DecisionEvent.create(
            event_type=event_type,
            deployment_pseudonym=prediction.deployment_pseudonym,
            decision_id=decision_id,
            decision_cutoff_utc=cutoff,
            event_time_utc=cutoff,
            recorded_at_utc=recorded,
            stream_sequence=len(events) + 1,
            causation_event_id=causation_event_id,
            correlation_id=correlation_id,
            payload=payload,
            previous_event_sha256=previous,
        )
        events.append(event)
        return event

    snapshot_event = append(
        DecisionEventType.SNAPSHOT_SEALED,
        SnapshotSealedPayload(
            snapshot_id=serving_snapshot.snapshot_id,
            snapshot_sha256=serving_snapshot.content_hash(),
            mode=serving_snapshot.mode,
            watermark_utc=serving_snapshot.as_of_ingestion_time_utc,
            source_digests={
                "operational_profile": prediction.operational_profile_hash,
                "telemetry": prediction.telemetry_snapshot_hash,
                "topology": prediction.topology_snapshot_hash,
            },
        ),
    )
    support_event = append(
        DecisionEventType.SUPPORT_ASSESSED,
        SupportAssessedPayload(
            state=_support_state(prediction),
            axes=dict(sorted(prediction.support_axes.items())),
            reason_codes=(
                ("frozen_v06_support",)
                if prediction.support_score is not None
                else ("support_score_unavailable",)
            ),
            profile_sha256=prediction.operational_profile_hash,
        ),
        causation_event_id=snapshot_event.event_id,
    )
    quality_event = append(
        DecisionEventType.DATA_QUALITY_ASSESSED,
        DataQualityAssessedPayload(
            state=prediction.data_quality_state,
            warning_codes=prediction.data_quality_warnings,
            expected_observation_count=(
                serving_snapshot.expected_observation_count
            ),
            available_observation_count=(
                serving_snapshot.available_observation_count
            ),
            applicability_summary={},
        ),
        causation_event_id=snapshot_event.event_id,
    )
    incident_event = append(
        DecisionEventType.INCIDENT_EVIDENCE_PRODUCED,
        IncidentEvidenceProducedPayload(prediction=prediction),
        causation_event_id=quality_event.event_id,
    )
    append(
        DecisionEventType.CANDIDATE_SET_PRODUCED,
        CandidateSetProducedPayload(
            candidate_set=candidate_context.candidate_set
        ),
        causation_event_id=support_event.event_id,
    )
    if prediction.stage2_top_k:
        append(
            DecisionEventType.RANKING_PRODUCED,
            RankingProducedPayload(
                ranking=prediction.stage2_top_k,
                ranker_identity_sha256=prediction.model_bundle_hash,
            ),
            causation_event_id=incident_event.event_id,
        )
    evidence = _evidence_set(
        prediction,
        incident_event_id=incident_event.event_id,
    )
    causal_event = append(
        DecisionEventType.CAUSAL_EVIDENCE_PRODUCED,
        CausalEvidenceProducedPayload(evidence_set=evidence),
        causation_event_id=incident_event.event_id,
    )
    append(
        DecisionEventType.DECISION_SEALED,
        DecisionSealedPayload(
            final_state=prediction.final_decision,
            activation_path=prediction.activation_path,
            prediction_envelope_sha256=canonical_hash(prediction),
            chain_head_before_seal=events[-1].event_sha256,
        ),
        causation_event_id=causal_event.event_id,
    )
    stream = tuple(events)
    seal_decision_stream(stream)
    return stream
