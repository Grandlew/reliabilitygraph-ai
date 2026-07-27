from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.decision_events import (
    DecisionEvent,
    DecisionEventType,
)


def _payload(event: DecisionEvent) -> dict:
    return event.model_dump(mode="json")


def test_decision_event_schema_is_deterministic() -> None:
    assert DecisionEvent.model_json_schema() == DecisionEvent.model_json_schema()


def test_event_envelope_rejects_extra_fields(decision_events) -> None:
    payload = _payload(decision_events[0])
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        DecisionEvent.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["decision_cutoff_utc", "event_time_utc", "recorded_at_utc"],
)
def test_event_envelope_rejects_naive_datetimes(
    decision_events,
    field: str,
) -> None:
    payload = _payload(decision_events[0])
    payload[field] = "2026-07-25T10:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        DecisionEvent.model_validate(payload)


def test_event_envelope_rejects_unknown_type(decision_events) -> None:
    payload = _payload(decision_events[0])
    payload["event_type"] = "made_up_event"
    with pytest.raises(ValidationError):
        DecisionEvent.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["payload_sha256", "previous_event_sha256", "event_sha256"],
)
def test_event_envelope_rejects_bad_digest_shape(
    decision_events,
    field: str,
) -> None:
    payload = _payload(decision_events[0])
    payload[field] = "not-a-digest"
    with pytest.raises(ValidationError):
        DecisionEvent.model_validate(payload)


def test_event_envelope_rejects_nonpositive_sequence(decision_events) -> None:
    payload = _payload(decision_events[0])
    payload["stream_sequence"] = 0
    with pytest.raises(ValidationError):
        DecisionEvent.model_validate(payload)


def test_event_envelope_rejects_recording_before_fact(decision_events) -> None:
    payload = _payload(decision_events[0])
    payload["recorded_at_utc"] = (
        decision_events[0].event_time_utc - timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(ValidationError, match="Recorded time"):
        DecisionEvent.model_validate(payload)


def test_original_event_rejects_future_information(decision_events) -> None:
    payload = _payload(decision_events[1])
    payload["event_time_utc"] = (
        decision_events[1].decision_cutoff_utc + timedelta(seconds=1)
    ).isoformat()
    payload["recorded_at_utc"] = (
        decision_events[1].decision_cutoff_utc + timedelta(seconds=2)
    ).isoformat()
    with pytest.raises(ValidationError, match="future evidence"):
        DecisionEvent.model_validate(payload)


def test_event_rejects_payload_type_mismatch(decision_events) -> None:
    payload = _payload(decision_events[0])
    payload["event_type"] = DecisionEventType.SUPPORT_ASSESSED.value
    with pytest.raises(ValidationError):
        DecisionEvent.model_validate(payload)


@pytest.mark.parametrize("location", ["schema_version", "payload"])
def test_event_rejects_unknown_schema_version(
    decision_events,
    location: str,
) -> None:
    payload = _payload(decision_events[0])
    if location == "payload":
        payload["payload"]["schema_version"] = "1.0.0"
    else:
        payload["schema_version"] = "1.0.0"
    with pytest.raises(ValidationError):
        DecisionEvent.model_validate(payload)


def test_event_rejects_tampered_payload(decision_events) -> None:
    payload = _payload(decision_events[1])
    payload["payload"]["reason_codes"] = ["tampered"]
    with pytest.raises(ValidationError, match="payload commitment"):
        DecisionEvent.model_validate(payload)


@pytest.mark.parametrize("event_index", range(8))
def test_every_original_variant_round_trips_canonical_json(
    decision_events,
    event_index: int,
) -> None:
    event = decision_events[event_index]
    assert DecisionEvent.model_validate_json(event.model_dump_json()) == event


def test_event_creation_is_idempotent(decision_events) -> None:
    event = decision_events[0]
    recreated = DecisionEvent.create(
        event_type=event.event_type,
        deployment_pseudonym=event.deployment_pseudonym,
        decision_id=event.decision_id,
        decision_cutoff_utc=event.decision_cutoff_utc,
        event_time_utc=event.event_time_utc,
        recorded_at_utc=event.recorded_at_utc,
        stream_sequence=event.stream_sequence,
        correlation_id=event.correlation_id,
        payload=event.payload,
        previous_event_sha256=event.previous_event_sha256,
    )
    assert recreated == event
