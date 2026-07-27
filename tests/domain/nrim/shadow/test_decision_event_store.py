from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from app.domain.nrim.shadow.decision_events import (
    AmendmentReason,
    DecisionEvent,
    DecisionEventType,
    EvidenceAmendedPayload,
)
from app.domain.nrim.shadow.decision_store import DecisionEventStore
from app.domain.nrim.shadow.hashing import canonical_hash
from app.domain.nrim.shadow.store import (
    AppendOnlyEvidenceStore,
    ConflictingDuplicateError,
)


def _store(tmp_path) -> DecisionEventStore:
    return DecisionEventStore(
        AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    )


def _amended(decision_events):
    target = decision_events[3]
    event = DecisionEvent.create(
        event_type=DecisionEventType.EVIDENCE_AMENDED,
        deployment_pseudonym=target.deployment_pseudonym,
        decision_id=target.decision_id,
        decision_cutoff_utc=target.decision_cutoff_utc,
        event_time_utc=target.decision_cutoff_utc + timedelta(minutes=5),
        recorded_at_utc=target.recorded_at_utc + timedelta(hours=1),
        stream_sequence=len(decision_events) + 1,
        causation_event_id=decision_events[-1].event_id,
        correlation_id=target.correlation_id,
        payload=EvidenceAmendedPayload(
            target_event_id=target.event_id,
            reason_code=AmendmentReason.LATE_OBSERVATION,
            evidence_references=("late_observation_1",),
            amendment_sha256=canonical_hash({"late": 1}),
        ),
        previous_event_sha256=decision_events[-1].event_sha256,
    )
    return (*decision_events, event)


def test_store_appends_whole_stream_and_reads_it(
    tmp_path,
    decision_events,
) -> None:
    store = _store(tmp_path)
    result = store.append_stream(decision_events)
    assert result.inserted
    assert result.inserted_event_count == len(decision_events)
    assert store.read_stream(result.decision_id) == decision_events


def test_identical_retry_is_idempotent(tmp_path, decision_events) -> None:
    store = _store(tmp_path)
    first = store.append_stream(decision_events)
    second = store.append_stream(decision_events)
    assert first.stream_sha256 == second.stream_sha256
    assert not second.inserted
    assert second.inserted_event_count == 0


def test_concurrent_identical_append_is_idempotent(
    tmp_path,
    decision_events,
) -> None:
    store = _store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _: store.append_stream(decision_events),
                range(2),
            )
        )
    assert sum(item.inserted for item in results) == 1
    assert store.read_stream(decision_events[0].decision_id) == decision_events


def test_conflicting_duplicate_fails_closed(
    tmp_path,
    decision_events,
) -> None:
    store = _store(tmp_path)
    store.append_stream(decision_events)
    conflict = decision_events[0].model_copy(
        update={"event_sha256": "0" * 64}
    )
    with pytest.raises((ValueError, ConflictingDuplicateError)):
        store.append_stream((conflict, *decision_events[1:]))
    assert store.read_stream(decision_events[0].decision_id) == decision_events


def test_partial_stream_is_never_committed(tmp_path, decision_events) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.append_stream(decision_events[:-1])
    with pytest.raises(KeyError):
        store.read_stream(decision_events[0].decision_id)


@pytest.mark.parametrize("statement", ["UPDATE", "DELETE"])
def test_database_triggers_reject_event_mutation(
    tmp_path,
    decision_events,
    statement: str,
) -> None:
    store = _store(tmp_path)
    store.append_stream(decision_events)
    with store.evidence_store._connection() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            if statement == "UPDATE":
                connection.execute(
                    "UPDATE decision_events SET event_type = 'changed'"
                )
            else:
                connection.execute("DELETE FROM decision_events")


def test_store_appends_late_events_without_rewriting_original(
    tmp_path,
    decision_events,
) -> None:
    store = _store(tmp_path)
    original = store.append_stream(decision_events)
    extended = store.append_stream(_amended(decision_events))
    assert extended.inserted_event_count == 1
    assert extended.stream_sha256 != original.stream_sha256
    assert store.read_stream(original.decision_id)[:8] == decision_events


def test_stream_listing_is_deterministic(tmp_path, decision_events) -> None:
    store = _store(tmp_path)
    store.append_stream(decision_events)
    references = store.list_streams(
        deployment_pseudonym=decision_events[0].deployment_pseudonym,
        cutoff_start_utc=decision_events[0].decision_cutoff_utc
        - timedelta(hours=1),
        cutoff_end_utc=decision_events[0].decision_cutoff_utc
        + timedelta(hours=1),
    )
    assert tuple(item.decision_id for item in references) == (
        decision_events[0].decision_id,
    )


def test_stream_listing_rejects_inverted_range(
    tmp_path,
    decision_events,
) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="inverted"):
        store.list_streams(
            deployment_pseudonym=decision_events[0].deployment_pseudonym,
            cutoff_start_utc=decision_events[0].decision_cutoff_utc,
            cutoff_end_utc=decision_events[0].decision_cutoff_utc
            - timedelta(seconds=1),
        )
