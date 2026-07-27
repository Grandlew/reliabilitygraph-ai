from __future__ import annotations

import pytest

from app.domain.nrim.shadow.compatibility_v06 import (
    CandidateContext,
    wrap_v06_prediction,
)
from app.domain.nrim.shadow.contracts import (
    DataQualityState,
    DecisionState,
    PredictionEnvelope,
    ServingSnapshot,
)
from app.domain.nrim.shadow.decision_envelope import seal_decision_stream
from app.domain.nrim.shadow.decision_events import (
    CandidateSetProducedPayload,
    CausalEvidenceProducedPayload,
    IncidentEvidenceProducedPayload,
    RankingProducedPayload,
)
from app.domain.nrim.shadow.hashing import canonical_hash


def _incident(events) -> IncidentEvidenceProducedPayload:
    return next(
        event.payload
        for event in events
        if isinstance(event.payload, IncidentEvidenceProducedPayload)
    )


def test_wrapper_preserves_every_prediction_field(
    decision_snapshot,
    decision_prediction,
    decision_context,
) -> None:
    events = wrap_v06_prediction(
        serving_snapshot=decision_snapshot,
        prediction=decision_prediction,
        candidate_context=decision_context,
    )
    assert _incident(events).prediction == decision_prediction
    assert canonical_hash(_incident(events).prediction) == canonical_hash(
        decision_prediction
    )


def test_wrapper_preserves_rank_order_and_precision(
    decision_events,
    decision_prediction,
) -> None:
    ranking = next(
        event.payload
        for event in decision_events
        if isinstance(event.payload, RankingProducedPayload)
    )
    assert ranking.ranking == decision_prediction.stage2_top_k


def test_wrapper_records_actual_candidate_universe(
    decision_events,
    decision_snapshot,
) -> None:
    candidates = next(
        event.payload.candidate_set
        for event in decision_events
        if isinstance(event.payload, CandidateSetProducedPayload)
    )
    assert set(candidates.included_candidates) == {
        item.component_pseudonym for item in decision_snapshot.components
    }


def test_ranked_nodes_must_belong_to_candidate_set(
    decision_snapshot,
    decision_prediction,
    decision_context,
) -> None:
    invalid = decision_context.model_copy(
        update={
            "candidate_set": decision_context.candidate_set.model_copy(
                update={"included_candidates": ()}
            )
        }
    )
    with pytest.raises(ValueError, match="outside"):
        wrap_v06_prediction(
            serving_snapshot=decision_snapshot,
            prediction=decision_prediction,
            candidate_context=invalid,
        )


def test_frozen_evidence_mapping_preserves_values(decision_events) -> None:
    causal = next(
        event.payload.evidence_set
        for event in decision_events
        if isinstance(event.payload, CausalEvidenceProducedPayload)
    )
    values = {
        item.mechanism_code: item.observed_value
        for item in causal.obligations
    }
    assert values["local_anomaly"] == 1.0
    assert values["topology_residual"] == -0.25


def test_frozen_evidence_mapping_records_alternatives(
    decision_events,
) -> None:
    causal = next(
        event.payload.evidence_set
        for event in decision_events
        if isinstance(event.payload, CausalEvidenceProducedPayload)
    )
    assert causal.alternatives
    assert all(
        item.candidate_pseudonym
        != item.alternative_candidate_pseudonym
        for item in causal.alternatives
    )


def test_unknown_route_emits_no_stage2(
    decision_snapshot,
    decision_prediction,
) -> None:
    payload = decision_prediction.model_dump(mode="json")
    payload.update(
        {
            "stage2_top_k": [],
            "final_decision": DecisionState.UNKNOWN.value,
            "episode_state_after": DecisionState.UNKNOWN.value,
            "dual_path_state": DecisionState.UNKNOWN.value,
            "activation_path": "unsupported",
        }
    )
    prediction = PredictionEnvelope.model_validate(payload)
    context = CandidateContext.from_v06(
        serving_snapshot=decision_snapshot,
        prediction=prediction,
    )
    events = wrap_v06_prediction(
        serving_snapshot=decision_snapshot,
        prediction=prediction,
        candidate_context=context,
    )
    assert not any(
        isinstance(event.payload, RankingProducedPayload)
        for event in events
    )
    seal_decision_stream(events)


def test_blocked_quality_emits_no_model_or_stage2_output(
    decision_snapshot,
    decision_prediction,
) -> None:
    snapshot_payload = decision_snapshot.model_dump(mode="json")
    snapshot_payload.update(
        {
            "observations": [],
            "data_quality_state": DataQualityState.BLOCKED.value,
            "data_quality_warnings": ["collector_outage"],
            "available_observation_count": 0,
        }
    )
    snapshot = ServingSnapshot.model_validate(snapshot_payload)
    prediction_payload = decision_prediction.model_dump(mode="json")
    prediction_payload.update(
        {
            "stage1_probability": None,
            "healthy_residual": None,
            "support_score": None,
            "impact_score": None,
            "stage2_top_k": [],
            "fast_path_state": None,
            "slow_path_state": None,
            "dual_path_state": None,
            "final_decision": DecisionState.DATA_QUALITY_ESCALATION.value,
            "episode_state_after": (
                DecisionState.DATA_QUALITY_ESCALATION.value
            ),
            "activation_path": "data_quality",
            "data_quality_state": DataQualityState.BLOCKED.value,
            "data_quality_warnings": ["collector_outage"],
        }
    )
    prediction = PredictionEnvelope.model_validate(prediction_payload)
    context = CandidateContext.from_v06(
        serving_snapshot=snapshot,
        prediction=prediction,
    )
    events = wrap_v06_prediction(
        serving_snapshot=snapshot,
        prediction=prediction,
        candidate_context=context,
    )
    assert _incident(events).prediction.stage1_probability is None
    assert not any(
        isinstance(event.payload, RankingProducedPayload)
        for event in events
    )
