from app.domain.nrim.diagnostic_ranker import (
    calculate_diagnostic_utility,
    create_catchup_diagnostic_options,
    rank_diagnostics,
)
from app.domain.nrim.reasoning import (
    DiagnosticOption,
    InterventionRisk,
)


def test_observation_test_ranks_above_high_risk_test() -> None:
    safe_test = DiagnosticOption(
        name="Inspect logs",
        question="What do the logs show?",
        hypothesis_ids=["hypothesis_1"],
        information_gain=0.8,
        hypothesis_separation=0.8,
        safety=1.0,
        reversibility=1.0,
        speed=0.9,
        low_cost=1.0,
        operational_risk=InterventionRisk.OBSERVE_ONLY,
    )

    dangerous_test = DiagnosticOption(
        name="Modify production system",
        question="Does a production change alter the failure?",
        hypothesis_ids=["hypothesis_1"],
        information_gain=1.0,
        hypothesis_separation=1.0,
        safety=0.1,
        reversibility=0.3,
        speed=0.5,
        low_cost=0.5,
        operational_risk=InterventionRisk.HIGH,
        requires_engineer=True,
    )

    assert calculate_diagnostic_utility(
        safe_test
    ) > calculate_diagnostic_utility(dangerous_test)


def test_ranker_filters_irrelevant_tests() -> None:
    options = create_catchup_diagnostic_options()

    ranked = rank_diagnostics(
        options,
        leading_hypothesis_ids={"hyp_storage_capacity"},
    )

    assert ranked
    assert all(
        "hyp_storage_capacity" in option.hypothesis_ids
        for option in ranked
    )
