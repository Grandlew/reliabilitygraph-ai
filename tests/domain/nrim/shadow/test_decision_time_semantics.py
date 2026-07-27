from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.decision_events import (
    AmendmentReason,
    DecisionEvent,
    DecisionEventType,
    EvidenceAmendedPayload,
    normalize_utc,
)
from app.domain.nrim.shadow.decision_projection import (
    DecisionSnapshot,
    project_decision,
)
from app.domain.nrim.shadow.hashing import canonical_hash


def _late(decision_events) -> DecisionEvent:
    target = decision_events[3]
    return DecisionEvent.create(
        event_type=DecisionEventType.EVIDENCE_AMENDED,
        deployment_pseudonym=target.deployment_pseudonym,
        decision_id=target.decision_id,
        decision_cutoff_utc=target.decision_cutoff_utc,
        event_time_utc=target.decision_cutoff_utc + timedelta(minutes=15),
        recorded_at_utc=target.recorded_at_utc + timedelta(hours=1),
        stream_sequence=9,
        causation_event_id=decision_events[-1].event_id,
        correlation_id=target.correlation_id,
        payload=EvidenceAmendedPayload(
            target_event_id=target.event_id,
            reason_code=AmendmentReason.LATE_OBSERVATION,
            evidence_references=("late_observation",),
            amendment_sha256=canonical_hash({"late": True}),
        ),
        previous_event_sha256=decision_events[-1].event_sha256,
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-25T13:00:00+03:00",
        "2026-07-25T05:00:00-05:00",
        "2026-07-25T10:00:00+00:00",
    ],
)
def test_utc_normalization_is_offset_independent(timestamp: str) -> None:
    assert normalize_utc(datetime.fromisoformat(timestamp)).isoformat() == (
        "2026-07-25T10:00:00+00:00"
    )


def test_utc_normalization_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        normalize_utc(datetime(2026, 7, 25, 10))


def test_original_event_allows_equal_cutoff_boundary(
    decision_events,
) -> None:
    assert all(
        item.event_time_utc == item.decision_cutoff_utc
        for item in decision_events
    )


def test_late_evidence_is_allowed_only_as_retrospective_event(
    decision_events,
) -> None:
    amendment = _late(decision_events)
    assert amendment.event_time_utc > amendment.decision_cutoff_utc
    original_payload = decision_events[1].model_dump(mode="json")
    original_payload["event_time_utc"] = amendment.event_time_utc.isoformat()
    original_payload["recorded_at_utc"] = amendment.recorded_at_utc.isoformat()
    with pytest.raises(ValidationError, match="future evidence"):
        DecisionEvent.model_validate(original_payload)


def test_as_known_boundary_includes_event_recorded_exactly_then(
    decision_events,
) -> None:
    amendment = _late(decision_events)
    snapshot = project_decision(
        (*decision_events, amendment),
        view="as_known_at",
        as_known_at_utc=amendment.recorded_at_utc,
    )
    assert isinstance(snapshot, DecisionSnapshot)
    assert len(snapshot.applied_amendments) == 1


def test_as_known_boundary_excludes_event_one_microsecond_early(
    decision_events,
) -> None:
    amendment = _late(decision_events)
    snapshot = project_decision(
        (*decision_events, amendment),
        view="as_known_at",
        as_known_at_utc=amendment.recorded_at_utc
        - timedelta(microseconds=1),
    )
    assert isinstance(snapshot, DecisionSnapshot)
    assert not snapshot.applied_amendments
