from __future__ import annotations

from datetime import datetime
from typing import Iterable, Literal

from pydantic import Field

from .candidate_evidence import (
    CandidateSet,
    CausalEvidenceSet,
    StrictDecisionModel,
)
from .contracts import DataQualityState, DecisionState, PredictionEnvelope, RankedCause
from .decision_envelope import seal_decision_stream
from .decision_events import (
    AdjudicationLinkedPayload,
    CandidateSetProducedPayload,
    CausalEvidenceProducedPayload,
    DataQualityAssessedPayload,
    DecisionEvent,
    DecisionEventType,
    DecisionSealedPayload,
    EvidenceAmendedPayload,
    IncidentEvidenceProducedPayload,
    SupportAssessedPayload,
    SupportState,
    normalize_utc,
)
from .hashing import typed_hash


DecisionView = Literal["original", "as_known_at", "latest", "comparison"]


class DecisionSnapshot(StrictDecisionModel):
    decision_id: str
    deployment_pseudonym: str
    decision_cutoff_utc: datetime
    view: Literal["original", "as_known_at", "latest"]
    as_known_at_utc: datetime | None = None
    support_state: SupportState
    support_axes: dict[str, float]
    support_reason_codes: tuple[str, ...]
    data_quality_state: DataQualityState
    data_quality_warning_codes: tuple[str, ...]
    prediction: PredictionEnvelope
    candidate_set: CandidateSet
    ranking: tuple[RankedCause, ...]
    causal_evidence: CausalEvidenceSet
    final_state: DecisionState
    activation_path: str
    original_event_sha256s: tuple[str, ...]
    applied_amendments: tuple[EvidenceAmendedPayload, ...] = ()
    adjudication_links: tuple[AdjudicationLinkedPayload, ...] = ()
    projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = "0.7.2"


class DecisionEventDelta(StrictDecisionModel):
    added_event_ids: tuple[str, ...]
    amendment_target_event_ids: tuple[str, ...]
    adjudication_ids: tuple[str, ...]


class DecisionComparison(StrictDecisionModel):
    original: DecisionSnapshot
    latest: DecisionSnapshot
    delta: DecisionEventDelta
    comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = "0.7.2"


def _payload(
    events: tuple[DecisionEvent, ...],
    event_type: DecisionEventType,
):
    matches = tuple(item for item in events if item.event_type is event_type)
    if len(matches) != 1:
        raise ValueError(f"Projection requires one {event_type.value}")
    return matches[0].payload


def _snapshot(
    *,
    original_events: tuple[DecisionEvent, ...],
    selected_events: tuple[DecisionEvent, ...],
    view: Literal["original", "as_known_at", "latest"],
    as_known_at_utc: datetime | None,
) -> DecisionSnapshot:
    support = _payload(original_events, DecisionEventType.SUPPORT_ASSESSED)
    quality = _payload(
        original_events,
        DecisionEventType.DATA_QUALITY_ASSESSED,
    )
    incident = _payload(
        original_events,
        DecisionEventType.INCIDENT_EVIDENCE_PRODUCED,
    )
    candidates = _payload(
        original_events,
        DecisionEventType.CANDIDATE_SET_PRODUCED,
    )
    causal = _payload(
        original_events,
        DecisionEventType.CAUSAL_EVIDENCE_PRODUCED,
    )
    decision = _payload(original_events, DecisionEventType.DECISION_SEALED)
    assert isinstance(support, SupportAssessedPayload)
    assert isinstance(quality, DataQualityAssessedPayload)
    assert isinstance(incident, IncidentEvidenceProducedPayload)
    assert isinstance(candidates, CandidateSetProducedPayload)
    assert isinstance(causal, CausalEvidenceProducedPayload)
    assert isinstance(decision, DecisionSealedPayload)
    amendments = tuple(
        item.payload
        for item in selected_events
        if isinstance(item.payload, EvidenceAmendedPayload)
    )
    adjudications = tuple(
        item.payload
        for item in selected_events
        if isinstance(item.payload, AdjudicationLinkedPayload)
    )
    core = {
        "decision_id": original_events[0].decision_id,
        "deployment_pseudonym": original_events[0].deployment_pseudonym,
        "decision_cutoff_utc": original_events[0].decision_cutoff_utc,
        "view": view,
        "as_known_at_utc": as_known_at_utc,
        "support_state": support.state,
        "support_axes": support.axes,
        "support_reason_codes": support.reason_codes,
        "data_quality_state": quality.state,
        "data_quality_warning_codes": quality.warning_codes,
        "prediction": incident.prediction,
        "candidate_set": candidates.candidate_set,
        "ranking": incident.prediction.stage2_top_k,
        "causal_evidence": causal.evidence_set,
        "final_state": decision.final_state,
        "activation_path": decision.activation_path,
        "original_event_sha256s": tuple(
            item.event_sha256 for item in original_events
        ),
        "applied_amendments": amendments,
        "adjudication_links": adjudications,
    }
    return DecisionSnapshot(
        **core,
        projection_sha256=typed_hash(
            type_name="decision_snapshot",
            schema_version="0.7.2",
            value=core,
        ),
    )


def project_decision(
    events: Iterable[DecisionEvent],
    *,
    view: DecisionView,
    as_known_at_utc: datetime | None = None,
) -> DecisionSnapshot | DecisionComparison:
    stream = tuple(
        sorted(events, key=lambda item: item.stream_sequence)
    )
    envelope = seal_decision_stream(stream)
    seal_index = next(
        index
        for index, item in enumerate(stream)
        if item.event_id == envelope.decision_seal_event_id
    )
    original_events = stream[: seal_index + 1]
    if view != "as_known_at" and as_known_at_utc is not None:
        raise ValueError("Knowledge time is only valid for as-known-at view")
    if view == "original":
        return _snapshot(
            original_events=original_events,
            selected_events=original_events,
            view="original",
            as_known_at_utc=None,
        )
    if view == "latest":
        return _snapshot(
            original_events=original_events,
            selected_events=stream,
            view="latest",
            as_known_at_utc=None,
        )
    if view == "as_known_at":
        if as_known_at_utc is None:
            raise ValueError("as_known_at_utc is required")
        knowledge_time = normalize_utc(as_known_at_utc)
        selected = tuple(
            item
            for item in stream
            if item.recorded_at_utc <= knowledge_time
        )
        if not any(
            item.event_type is DecisionEventType.DECISION_SEALED
            for item in selected
        ):
            raise ValueError("Decision was not sealed at requested knowledge time")
        return _snapshot(
            original_events=original_events,
            selected_events=selected,
            view="as_known_at",
            as_known_at_utc=knowledge_time,
        )
    if view == "comparison":
        original = _snapshot(
            original_events=original_events,
            selected_events=original_events,
            view="original",
            as_known_at_utc=None,
        )
        latest = _snapshot(
            original_events=original_events,
            selected_events=stream,
            view="latest",
            as_known_at_utc=None,
        )
        added = stream[seal_index + 1 :]
        delta = DecisionEventDelta(
            added_event_ids=tuple(item.event_id for item in added),
            amendment_target_event_ids=tuple(
                item.payload.target_event_id
                for item in added
                if isinstance(item.payload, EvidenceAmendedPayload)
            ),
            adjudication_ids=tuple(
                item.payload.adjudication_id
                for item in added
                if isinstance(item.payload, AdjudicationLinkedPayload)
            ),
        )
        core = {
            "original": original,
            "latest": latest,
            "delta": delta,
        }
        return DecisionComparison(
            **core,
            comparison_sha256=typed_hash(
                type_name="decision_comparison",
                schema_version="0.7.2",
                value=core,
            ),
        )
    raise ValueError(f"Unsupported decision view: {view!r}")
