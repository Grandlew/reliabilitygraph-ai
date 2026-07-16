import pytest
from pydantic import ValidationError

from app.domain.nrim.reasoning import (
    EvidenceRole,
    EvidenceStrength,
    HypothesisEvidenceLink,
    InterventionRisk,
    ReasoningEvidence,
    RecommendedIntervention,
)
from app.domain.nrim.telemetry import TelemetryQuality


def test_quarantined_evidence_cannot_have_nonzero_weight() -> None:
    evidence = ReasoningEvidence(
        evidence_id="evidence_1",
        statement="Invalid telemetry evidence.",
        role=EvidenceRole.SUPPORTS,
        strength=EvidenceStrength.STRONG,
        quality=TelemetryQuality.QUARANTINED,
        independent_source="test",
    )

    with pytest.raises(ValidationError):
        HypothesisEvidenceLink(
            hypothesis_id="hypothesis_1",
            evidence=evidence,
            weight=4.0,
        )


def test_prohibited_intervention_requires_engineer() -> None:
    with pytest.raises(ValidationError):
        RecommendedIntervention(
            name="Modify production network",
            hypothesis_id="hypothesis_1",
            target_node_ids=["switch_1"],
            rationale="Test intervention.",
            expected_effect="Change network behaviour.",
            operational_risk=(
                InterventionRisk.PROHIBITED_AUTONOMOUSLY
            ),
            reversible=True,
            rollback_plan="Restore previous network configuration.",
            requires_engineer_approval=False,
            verification_test="Verify network service.",
        )
