from app.domain.nrim.hypothesis_engine import (
    create_catchup_storage_hypotheses,
)
from app.domain.nrim.intervention_engine import (
    create_interventions,
)
from app.domain.nrim.reasoning import HypothesisStatus


def test_no_intervention_for_unresolved_hypothesis() -> None:
    hypothesis = create_catchup_storage_hypotheses()[0]

    interventions = create_interventions(hypothesis)

    assert interventions == []


def test_supported_hypothesis_gets_safe_first_action() -> None:
    hypothesis = create_catchup_storage_hypotheses()[0]

    hypothesis = hypothesis.model_copy(
        update={
            "status": HypothesisStatus.SUPPORTED,
        }
    )

    interventions = create_interventions(hypothesis)

    assert interventions
    assert interventions[0].requires_engineer_approval is False
