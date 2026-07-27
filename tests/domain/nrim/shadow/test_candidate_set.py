from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.candidate_evidence import (
    CandidateCoverage,
    CandidateExclusion,
    CandidateExclusionReason,
    CandidateSet,
)


def test_candidate_set_is_canonical_and_digest_bound(decision_context) -> None:
    candidate_set = decision_context.candidate_set
    assert candidate_set.included_candidates == tuple(
        sorted(candidate_set.included_candidates)
    )
    assert len(candidate_set.digest) == 64


def test_candidate_set_rejects_duplicate_inclusions(decision_context) -> None:
    payload = decision_context.candidate_set.model_dump(mode="json")
    payload["included_candidates"] = [
        payload["included_candidates"][0],
        payload["included_candidates"][0],
    ]
    with pytest.raises(ValidationError, match="duplicate"):
        CandidateSet.model_validate(payload)


def test_candidate_set_rejects_nondeterministic_order(
    decision_context,
) -> None:
    payload = decision_context.candidate_set.model_dump(mode="json")
    payload["included_candidates"] = list(
        reversed(payload["included_candidates"])
    )
    with pytest.raises(ValidationError, match="canonical"):
        CandidateSet.model_validate(payload)


def test_candidate_set_rejects_included_excluded_overlap(
    decision_context,
) -> None:
    payload = decision_context.candidate_set.model_dump(mode="json")
    payload["exclusions"] = [
        CandidateExclusion(
            component_pseudonym=payload["included_candidates"][0],
            reason=CandidateExclusionReason.POLICY_EXCLUDED,
            rationale_code="test_policy",
        ).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="overlap"):
        CandidateSet.model_validate(payload)


def test_candidate_set_rejects_missing_generator_identity(
    decision_context,
) -> None:
    payload = decision_context.candidate_set.model_dump(mode="json")
    payload.pop("generator")
    with pytest.raises(ValidationError):
        CandidateSet.model_validate(payload)


@pytest.mark.parametrize(
    ("eligible", "included", "denominator"),
    [(1, 2, 1), (2, 1, 1), (0, 0, 0)],
)
def test_candidate_coverage_rejects_invalid_denominators(
    eligible: int,
    included: int,
    denominator: int,
) -> None:
    with pytest.raises(ValidationError):
        CandidateCoverage(
            eligible_true_cause_count=eligible,
            included_true_cause_count=included,
            denominator=denominator,
            adjudication_reference="adjudication-v1",
        )


def test_excluded_true_cause_is_explicitly_measurable() -> None:
    coverage = CandidateCoverage(
        eligible_true_cause_count=1,
        included_true_cause_count=0,
        denominator=1,
        adjudication_reference="adjudication-v1",
    )
    assert coverage.included_true_cause_count == 0
    assert coverage.denominator == 1


def test_candidate_coverage_is_absent_without_adjudication(
    decision_context,
) -> None:
    assert decision_context.candidate_set.coverage is None


def test_candidate_set_normalizes_cutoff_to_utc(decision_context) -> None:
    payload = decision_context.candidate_set.model_dump(mode="json")
    payload["decision_cutoff_utc"] = "2026-07-25T13:00:00+03:00"
    candidate_set = CandidateSet.model_validate(payload)
    assert candidate_set.decision_cutoff_utc.isoformat() == (
        "2026-07-25T10:00:00+00:00"
    )


def test_candidate_set_rejects_naive_cutoff(decision_context) -> None:
    payload = decision_context.candidate_set.model_dump(mode="json")
    payload["decision_cutoff_utc"] = datetime(2026, 7, 25, 10)
    with pytest.raises(ValidationError, match="timezone"):
        CandidateSet.model_validate(payload)
