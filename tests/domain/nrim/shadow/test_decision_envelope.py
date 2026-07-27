from __future__ import annotations

import pytest

from app.domain.nrim.shadow.candidate_evidence import CausalEvidenceSet
from app.domain.nrim.shadow.decision_envelope import seal_decision_stream
from app.domain.nrim.shadow.decision_events import (
    CausalEvidenceProducedPayload,
    DecisionEvent,
    DecisionSealedPayload,
    SupportAssessedPayload,
    SupportState,
    ZERO_SHA256,
)


def _rebuild(decision_events, replacements: dict[int, object]):
    rebuilt = []
    id_map: dict[str, str] = {}
    for index, old in enumerate(decision_events):
        payload = replacements.get(index, old.payload)
        if isinstance(payload, CausalEvidenceProducedPayload):
            obligations = tuple(
                item.model_copy(
                    update={
                        "source_event_ids": tuple(
                            id_map.get(source, source)
                            for source in item.source_event_ids
                        )
                    }
                )
                for item in payload.evidence_set.obligations
            )
            payload = payload.model_copy(
                update={
                    "evidence_set": payload.evidence_set.model_copy(
                        update={"obligations": obligations}
                    )
                }
            )
        if isinstance(payload, DecisionSealedPayload):
            payload = payload.model_copy(
                update={
                    "chain_head_before_seal": (
                        rebuilt[-1].event_sha256
                        if rebuilt
                        else ZERO_SHA256
                    )
                }
            )
        event = DecisionEvent.create(
            event_type=old.event_type,
            deployment_pseudonym=old.deployment_pseudonym,
            decision_id=old.decision_id,
            decision_cutoff_utc=old.decision_cutoff_utc,
            event_time_utc=old.event_time_utc,
            recorded_at_utc=old.recorded_at_utc,
            stream_sequence=old.stream_sequence,
            correlation_id=old.correlation_id,
            causation_event_id=(
                id_map.get(old.causation_event_id)
                if old.causation_event_id
                else None
            ),
            payload=payload,
            previous_event_sha256=(
                rebuilt[-1].event_sha256 if rebuilt else ZERO_SHA256
            ),
        )
        rebuilt.append(event)
        id_map[old.event_id] = event.event_id
    return tuple(rebuilt)


def test_complete_stream_seals_deterministically(decision_events) -> None:
    first = seal_decision_stream(decision_events)
    second = seal_decision_stream(tuple(decision_events))
    assert first == second
    assert first.event_count == 8


@pytest.mark.parametrize("deleted_index", range(7))
def test_stream_rejects_missing_required_events(
    decision_events,
    deleted_index: int,
) -> None:
    stream = tuple(
        event
        for index, event in enumerate(decision_events)
        if index != deleted_index
    )
    with pytest.raises(ValueError):
        seal_decision_stream(stream)


def test_stream_rejects_duplicate_event(decision_events) -> None:
    stream = (decision_events[0], decision_events[0], *decision_events[1:])
    with pytest.raises(ValueError):
        seal_decision_stream(stream)


def test_stream_rejects_reordered_events(decision_events) -> None:
    stream = (
        decision_events[1],
        decision_events[0],
        *decision_events[2:],
    )
    with pytest.raises(ValueError, match="sequence"):
        seal_decision_stream(stream)


def test_stream_rejects_cross_splice(decision_events) -> None:
    foreign = decision_events[-1].model_copy(
        update={"decision_id": "decision_" + "z" * 32}
    )
    with pytest.raises(ValueError):
        seal_decision_stream((*decision_events[:-1], foreign))


def test_stream_rejects_broken_predecessor(decision_events) -> None:
    event = decision_events[3].model_copy(
        update={"previous_event_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="predecessor"):
        seal_decision_stream(
            (*decision_events[:3], event, *decision_events[4:])
        )


def test_stream_rejects_unsupported_ranking(decision_events) -> None:
    support = decision_events[1].payload
    assert isinstance(support, SupportAssessedPayload)
    rebuilt = _rebuild(
        decision_events,
        {1: support.model_copy(update={"state": SupportState.UNSUPPORTED})},
    )
    with pytest.raises(ValueError, match="cannot rank"):
        seal_decision_stream(rebuilt)


def test_stream_rejects_changed_activation_path(decision_events) -> None:
    sealed = decision_events[-1].payload
    assert isinstance(sealed, DecisionSealedPayload)
    rebuilt = _rebuild(
        decision_events,
        {
            7: sealed.model_copy(
                update={"activation_path": "rewritten"}
            )
        },
    )
    with pytest.raises(ValueError, match="activation"):
        seal_decision_stream(rebuilt)


def test_stream_rejects_rank_outside_candidate_set(decision_events) -> None:
    candidates = decision_events[4].payload
    candidate_set = candidates.candidate_set.model_copy(
        update={
            "included_candidates": (
                candidates.candidate_set.included_candidates[0],
            )
        }
    )
    rebuilt = _rebuild(
        decision_events,
        {4: candidates.model_copy(update={"candidate_set": candidate_set})},
    )
    with pytest.raises(ValueError, match="absent"):
        seal_decision_stream(rebuilt)
