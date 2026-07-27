from __future__ import annotations

from datetime import timedelta

import pytest

from app.domain.nrim.shadow.decision_events import (
    AdjudicationLinkedPayload,
    AmendmentReason,
    DecisionEvent,
    DecisionEventType,
    EvidenceAmendedPayload,
)
from app.domain.nrim.shadow.decision_projection import (
    DecisionComparison,
    DecisionSnapshot,
    project_decision,
)
from app.domain.nrim.shadow.hashing import canonical_hash


def _with_amendment(decision_events):
    original = decision_events
    target = original[3]
    amendment = DecisionEvent.create(
        event_type=DecisionEventType.EVIDENCE_AMENDED,
        deployment_pseudonym=target.deployment_pseudonym,
        decision_id=target.decision_id,
        decision_cutoff_utc=target.decision_cutoff_utc,
        event_time_utc=target.decision_cutoff_utc + timedelta(minutes=30),
        recorded_at_utc=target.recorded_at_utc + timedelta(hours=1),
        stream_sequence=len(original) + 1,
        causation_event_id=original[-1].event_id,
        correlation_id=target.correlation_id,
        payload=EvidenceAmendedPayload(
            target_event_id=target.event_id,
            reason_code=AmendmentReason.LATE_OBSERVATION,
            evidence_references=("observation_late_001",),
            amendment_sha256=canonical_hash({"late": "observation"}),
        ),
        previous_event_sha256=original[-1].event_sha256,
    )
    adjudication = DecisionEvent.create(
        event_type=DecisionEventType.ADJUDICATION_LINKED,
        deployment_pseudonym=target.deployment_pseudonym,
        decision_id=target.decision_id,
        decision_cutoff_utc=target.decision_cutoff_utc,
        event_time_utc=target.decision_cutoff_utc + timedelta(hours=1),
        recorded_at_utc=target.recorded_at_utc + timedelta(hours=2),
        stream_sequence=len(original) + 2,
        causation_event_id=amendment.event_id,
        correlation_id=target.correlation_id,
        payload=AdjudicationLinkedPayload(
            adjudication_id="adjudication_decision_001",
            adjudication_version=1,
            reviewed_decision_ids=(target.decision_id,),
            adjudication_sha256=canonical_hash({"outcome": "probable"}),
        ),
        previous_event_sha256=amendment.event_sha256,
    )
    return (*original, amendment, adjudication)


def test_original_projection_is_byte_stable(decision_events) -> None:
    first = project_decision(decision_events, view="original")
    second = project_decision(decision_events, view="original")
    assert first == second
    assert isinstance(first, DecisionSnapshot)


def test_projection_is_independent_of_input_iteration_order(
    decision_events,
) -> None:
    assert project_decision(
        tuple(reversed(decision_events)),
        view="original",
    ) == project_decision(decision_events, view="original")


def test_projection_rejects_incomplete_stream(decision_events) -> None:
    with pytest.raises(ValueError):
        project_decision(decision_events[:-1], view="original")


def test_original_ignores_late_amendments(decision_events) -> None:
    amended = _with_amendment(decision_events)
    before = project_decision(decision_events, view="original")
    after = project_decision(amended, view="original")
    assert before == after


def test_latest_exposes_amendments_without_rewriting_prediction(
    decision_events,
) -> None:
    amended = _with_amendment(decision_events)
    original = project_decision(amended, view="original")
    latest = project_decision(amended, view="latest")
    assert isinstance(original, DecisionSnapshot)
    assert isinstance(latest, DecisionSnapshot)
    assert latest.prediction == original.prediction
    assert len(latest.applied_amendments) == 1
    assert len(latest.adjudication_links) == 1


def test_as_known_at_uses_inclusive_recorded_time(decision_events) -> None:
    amended = _with_amendment(decision_events)
    at_amendment = amended[-2].recorded_at_utc
    snapshot = project_decision(
        amended,
        view="as_known_at",
        as_known_at_utc=at_amendment,
    )
    assert isinstance(snapshot, DecisionSnapshot)
    assert len(snapshot.applied_amendments) == 1
    assert not snapshot.adjudication_links


def test_as_known_at_excludes_future_recorded_events(decision_events) -> None:
    amended = _with_amendment(decision_events)
    before_amendment = amended[-2].recorded_at_utc - timedelta(seconds=1)
    snapshot = project_decision(
        amended,
        view="as_known_at",
        as_known_at_utc=before_amendment,
    )
    assert isinstance(snapshot, DecisionSnapshot)
    assert not snapshot.applied_amendments


def test_as_known_at_rejects_time_before_original_seal(
    decision_events,
) -> None:
    with pytest.raises(ValueError, match="not sealed"):
        project_decision(
            decision_events,
            view="as_known_at",
            as_known_at_utc=decision_events[0].recorded_at_utc
            - timedelta(seconds=1),
        )


def test_as_known_at_requires_aware_time(decision_events) -> None:
    naive = decision_events[-1].recorded_at_utc.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        project_decision(
            decision_events,
            view="as_known_at",
            as_known_at_utc=naive,
        )


def test_comparison_reports_event_level_delta(decision_events) -> None:
    comparison = project_decision(
        _with_amendment(decision_events),
        view="comparison",
    )
    assert isinstance(comparison, DecisionComparison)
    assert len(comparison.delta.added_event_ids) == 2
    assert comparison.original.prediction == comparison.latest.prediction
    assert comparison.delta.adjudication_ids == (
        "adjudication_decision_001",
    )


def test_knowledge_time_is_rejected_for_other_views(decision_events) -> None:
    with pytest.raises(ValueError, match="only valid"):
        project_decision(
            decision_events,
            view="latest",
            as_known_at_utc=decision_events[-1].recorded_at_utc,
        )
