from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.candidate_evidence import (
    CausalEvidenceSet,
    EvidenceApplicability,
    EvidenceObligation,
    EvidenceStatus,
)
from app.domain.nrim.shadow.decision_events import (
    CausalEvidenceProducedPayload,
)


def _evidence(decision_events) -> CausalEvidenceSet:
    payload = next(
        event.payload
        for event in decision_events
        if isinstance(event.payload, CausalEvidenceProducedPayload)
    )
    return payload.evidence_set


def test_every_ranked_candidate_has_evidence(decision_events) -> None:
    evidence = _evidence(decision_events)
    represented = {
        item.candidate_pseudonym for item in evidence.obligations
    }
    assert represented == set(evidence.ranked_candidates)


def test_support_and_contradiction_remain_distinct(decision_events) -> None:
    statuses = {item.status for item in _evidence(decision_events).obligations}
    assert EvidenceStatus.SUPPORTS in statuses
    assert EvidenceStatus.CONTRADICTS in statuses


def test_evidence_rejects_reference_outside_decision_stream(
    decision_events,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        _evidence(decision_events).validate_event_references(
            {"decision_event_" + "0" * 32}
        )


@pytest.mark.parametrize(
    ("applicability", "status"),
    [
        (EvidenceApplicability.NOT_APPLICABLE, EvidenceStatus.SUPPORTS),
        (EvidenceApplicability.APPLICABLE, EvidenceStatus.NOT_APPLICABLE),
    ],
)
def test_evidence_rejects_inconsistent_applicability_status(
    decision_events,
    applicability: EvidenceApplicability,
    status: EvidenceStatus,
) -> None:
    item = _evidence(decision_events).obligations[0]
    payload = item.model_dump(mode="json")
    payload["applicability"] = applicability.value
    payload["status"] = status.value
    with pytest.raises(ValidationError):
        EvidenceObligation.model_validate(payload)


@pytest.mark.parametrize(
    "status",
    [EvidenceStatus.SUPPORTS, EvidenceStatus.CONTRADICTS],
)
def test_observed_evidence_requires_immutable_reference(
    decision_events,
    status: EvidenceStatus,
) -> None:
    item = _evidence(decision_events).obligations[0]
    payload = item.model_dump(mode="json")
    payload["status"] = status.value
    payload["source_event_ids"] = []
    with pytest.raises(ValidationError, match="references"):
        EvidenceObligation.model_validate(payload)


def test_evidence_rejects_missing_mechanism(decision_events) -> None:
    item = _evidence(decision_events).obligations[0]
    payload = item.model_dump(mode="json")
    payload["mechanism_code"] = ""
    with pytest.raises(ValidationError):
        EvidenceObligation.model_validate(payload)


def test_ambiguous_candidates_require_alternatives(decision_events) -> None:
    evidence = _evidence(decision_events)
    payload = evidence.model_dump(mode="json")
    payload["ambiguous"] = True
    payload["alternatives"] = []
    with pytest.raises(ValidationError, match="alternative"):
        CausalEvidenceSet.model_validate(payload)


def test_alternatives_reference_known_obligations(decision_events) -> None:
    evidence = _evidence(decision_events)
    payload = evidence.model_dump(mode="json")
    payload["alternatives"][0]["discriminator_obligation_ids"] = [
        "obligation_" + "0" * 32
    ]
    with pytest.raises(ValidationError, match="unknown obligations"):
        CausalEvidenceSet.model_validate(payload)


def test_evidence_contract_has_no_causal_proof_boolean(
    decision_events,
) -> None:
    dumped = _evidence(decision_events).model_dump(mode="json")
    assert "causal_proof" not in dumped
    assert "proven_cause" not in dumped


def test_evidence_digest_is_deterministic(decision_events) -> None:
    evidence = _evidence(decision_events)
    assert evidence.digest == CausalEvidenceSet.model_validate(
        evidence.model_dump(mode="json")
    ).digest
