from __future__ import annotations

from typing import Iterable

from pydantic import Field

from .candidate_evidence import StrictDecisionModel
from .contracts import DataQualityState, DecisionState
from .decision_events import (
    CandidateSetProducedPayload,
    CausalEvidenceProducedPayload,
    DataQualityAssessedPayload,
    DecisionEvent,
    DecisionEventType,
    DecisionSealedPayload,
    EvidenceAmendedPayload,
    IncidentEvidenceProducedPayload,
    RankingProducedPayload,
    SupportAssessedPayload,
    SupportState,
    ZERO_SHA256,
)
from .hashing import typed_hash


class DecisionStreamEnvelope(StrictDecisionModel):
    decision_id: str
    deployment_pseudonym: str
    event_count: int = Field(gt=0)
    original_event_count: int = Field(gt=0)
    first_event_sha256: str
    original_chain_head_sha256: str
    latest_chain_head_sha256: str
    decision_seal_event_id: str
    stream_sha256: str
    schema_version: str = "0.7.2"


_REQUIRED_ORIGINAL = {
    DecisionEventType.SNAPSHOT_SEALED,
    DecisionEventType.SUPPORT_ASSESSED,
    DecisionEventType.DATA_QUALITY_ASSESSED,
    DecisionEventType.INCIDENT_EVIDENCE_PRODUCED,
    DecisionEventType.CANDIDATE_SET_PRODUCED,
    DecisionEventType.CAUSAL_EVIDENCE_PRODUCED,
    DecisionEventType.DECISION_SEALED,
}
_RETROSPECTIVE = {
    DecisionEventType.EVIDENCE_AMENDED,
    DecisionEventType.ADJUDICATION_LINKED,
}


def _one(
    events: tuple[DecisionEvent, ...],
    event_type: DecisionEventType,
) -> DecisionEvent:
    matches = tuple(item for item in events if item.event_type is event_type)
    if len(matches) != 1:
        raise ValueError(
            f"Decision stream requires exactly one {event_type.value}"
        )
    return matches[0]


def seal_decision_stream(
    events: Iterable[DecisionEvent],
) -> DecisionStreamEnvelope:
    stream = tuple(events)
    if not stream:
        raise ValueError("Decision stream cannot be empty")
    if tuple(item.stream_sequence for item in stream) != tuple(
        range(1, len(stream) + 1)
    ):
        raise ValueError("Decision stream sequence must be contiguous")
    identity = {
        (
            item.decision_id,
            item.deployment_pseudonym,
            item.decision_cutoff_utc,
            item.correlation_id,
        )
        for item in stream
    }
    if len(identity) != 1:
        raise ValueError("Decision stream contains mixed identities")
    previous = ZERO_SHA256
    event_ids: set[str] = set()
    for event in stream:
        if event.event_id in event_ids:
            raise ValueError("Decision stream contains duplicate events")
        if event.previous_event_sha256 != previous:
            raise ValueError("Decision stream predecessor chain is invalid")
        if (
            event.causation_event_id is not None
            and event.causation_event_id not in event_ids
        ):
            raise ValueError("Decision event has unavailable causation")
        event_ids.add(event.event_id)
        previous = event.event_sha256

    seal_event = _one(stream, DecisionEventType.DECISION_SEALED)
    seal_index = stream.index(seal_event)
    original = stream[: seal_index + 1]
    if any(item.event_type in _RETROSPECTIVE for item in original):
        raise ValueError("Retrospective evidence precedes the original seal")
    if any(item.event_type not in _RETROSPECTIVE for item in stream[seal_index + 1 :]):
        raise ValueError("Original decision events cannot follow the seal")
    original_types = {item.event_type for item in original}
    if not _REQUIRED_ORIGINAL.issubset(original_types):
        missing = sorted(
            item.value for item in _REQUIRED_ORIGINAL - original_types
        )
        raise ValueError("Decision stream is incomplete: " + ", ".join(missing))
    for event_type in _REQUIRED_ORIGINAL:
        _one(original, event_type)

    support_event = _one(original, DecisionEventType.SUPPORT_ASSESSED)
    quality_event = _one(
        original,
        DecisionEventType.DATA_QUALITY_ASSESSED,
    )
    incident_event = _one(
        original,
        DecisionEventType.INCIDENT_EVIDENCE_PRODUCED,
    )
    candidates_event = _one(
        original,
        DecisionEventType.CANDIDATE_SET_PRODUCED,
    )
    causal_event = _one(
        original,
        DecisionEventType.CAUSAL_EVIDENCE_PRODUCED,
    )
    support = support_event.payload
    quality = quality_event.payload
    incident = incident_event.payload
    candidates = candidates_event.payload
    causal = causal_event.payload
    decision = seal_event.payload
    assert isinstance(support, SupportAssessedPayload)
    assert isinstance(quality, DataQualityAssessedPayload)
    assert isinstance(incident, IncidentEvidenceProducedPayload)
    assert isinstance(candidates, CandidateSetProducedPayload)
    assert isinstance(causal, CausalEvidenceProducedPayload)
    assert isinstance(decision, DecisionSealedPayload)

    ranking_events = tuple(
        item
        for item in original
        if item.event_type is DecisionEventType.RANKING_PRODUCED
    )
    expected_ranking = incident.prediction.stage2_top_k
    if expected_ranking:
        if len(ranking_events) != 1:
            raise ValueError("Stage 2 output requires one ranking event")
        ranking_payload = ranking_events[0].payload
        assert isinstance(ranking_payload, RankingProducedPayload)
        if ranking_payload.ranking != expected_ranking:
            raise ValueError("Ranking event differs from frozen prediction")
        ranked_ids = tuple(
            item.component_pseudonym for item in ranking_payload.ranking
        )
        if not set(ranked_ids).issubset(
            candidates.candidate_set.included_candidates
        ):
            raise ValueError("A ranked component is absent from CandidateSet")
        if causal.evidence_set.ranked_candidates != ranked_ids:
            raise ValueError("Causal evidence differs from ranking order")
    else:
        if ranking_events:
            raise ValueError("Ranking exists without frozen Stage 2 output")
        if causal.evidence_set.ranked_candidates:
            raise ValueError("Causal evidence exists without Stage 2 output")

    blocked = (
        quality.state is DataQualityState.BLOCKED
        or support.state is SupportState.UNSUPPORTED
        or incident.prediction.final_decision
        in {
            DecisionState.UNKNOWN,
            DecisionState.ESCALATE,
            DecisionState.DATA_QUALITY_ESCALATION,
        }
    )
    if blocked and ranking_events:
        raise ValueError("Blocked or unsupported decisions cannot rank")
    if candidates.candidate_set.decision_cutoff_utc != (
        stream[0].decision_cutoff_utc
    ):
        raise ValueError("CandidateSet uses another decision cutoff")
    causal_index = original.index(causal_event)
    causal.evidence_set.validate_event_references(
        {item.event_id for item in original[:causal_index]}
    )
    if decision.final_state is not incident.prediction.final_decision:
        raise ValueError("Decision seal changes the frozen final state")
    if decision.activation_path != incident.prediction.activation_path:
        raise ValueError("Decision seal changes the activation path")

    original_ids = {item.event_id for item in original}
    for event in stream[seal_index + 1 :]:
        if isinstance(event.payload, EvidenceAmendedPayload):
            if event.payload.target_event_id not in original_ids:
                raise ValueError("Amendment targets another decision stream")

    stream_sha256 = typed_hash(
        type_name="decision_stream",
        schema_version=stream[0].schema_version,
        value=[item.event_sha256 for item in stream],
    )
    return DecisionStreamEnvelope(
        decision_id=stream[0].decision_id,
        deployment_pseudonym=stream[0].deployment_pseudonym,
        event_count=len(stream),
        original_event_count=len(original),
        first_event_sha256=stream[0].event_sha256,
        original_chain_head_sha256=seal_event.event_sha256,
        latest_chain_head_sha256=stream[-1].event_sha256,
        decision_seal_event_id=seal_event.event_id,
        stream_sha256=stream_sha256,
    )
