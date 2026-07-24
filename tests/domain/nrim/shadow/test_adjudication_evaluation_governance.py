from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.nrim.shadow.adjudication import (
    AdjudicationService,
    ReviewWorkflowError,
)
from app.domain.nrim.shadow.bundle import generate_signing_keypair
from app.domain.nrim.shadow.contracts import (
    AdjudicationConfidence,
    DataQualityState,
    DecisionState,
    IncidentAdjudication,
    IncidentSeverity,
    PredictionEnvelope,
    RankedCause,
    ReviewAnswer,
    SnapshotMode,
)
from app.domain.nrim.shadow.evaluation import (
    HealthyExposure,
    IncidentCase,
    ProspectiveMetricsEngine,
    cohen_kappa,
    exact_poisson_rate_upper,
)
from app.domain.nrim.shadow.governance import (
    default_protocol,
    register_protocol,
    verify_registered_protocol,
)
from app.domain.nrim.shadow.hashing import bytes_hash
from app.domain.nrim.shadow.release import (
    EvidenceOrigin,
    PilotContext,
    PromotionState,
    evaluate_release,
)
from app.domain.nrim.shadow.review_api import (
    ReviewPrincipal,
    TokenAuthorizer,
    create_review_router,
)
from app.domain.nrim.shadow.adjudication import ReviewerRole
from app.domain.nrim.shadow.store import AppendOnlyEvidenceStore


UTC = timezone.utc
START = datetime(2026, 8, 1, tzinfo=UTC)
DEPLOYMENT = "deployment_" + "a" * 40
ROOT = "component_" + "b" * 40
REVIEWER_A = "reviewer_" + "c" * 40
REVIEWER_B = "reviewer_" + "d" * 40
RESOLVER = "reviewer_" + "e" * 40


def adjudication(
    *,
    adjudication_id: str,
    reviewer: str,
    version: int,
    blinded: bool,
) -> IncidentAdjudication:
    return IncidentAdjudication(
        adjudication_id=adjudication_id,
        deployment_pseudonym=DEPLOYMENT,
        review_window_start_utc=START,
        review_window_end_utc=START + timedelta(hours=4),
        customer_impact_present=ReviewAnswer.YES,
        observable_degradation_present=ReviewAnswer.YES,
        impact_onset_utc=START + timedelta(hours=1),
        recovery_utc=START + timedelta(hours=3),
        severity=IncidentSeverity.HIGH,
        confirmed_root_component=ROOT,
        root_cause_family="storage",
        confidence=AdjudicationConfidence.CONFIRMED,
        existing_monitor_detected_first=False,
        nrim_top1_useful=(None if blinded else True),
        nrim_top3_useful=(None if blinded else True),
        would_change_investigation_order=(None if blinded else True),
        reviewer_id_pseudonym=reviewer,
        blinded_initial_assessment=blinded,
        evidence_notes="Observable service degradation and storage evidence.",
        evidence_note_source="engineer investigation",
        onset_time_source="service telemetry",
        onset_time_confidence=AdjudicationConfidence.CONFIRMED,
        adjudication_version=version,
        created_at_utc=START + timedelta(hours=5, minutes=version),
    )


def prediction(cutoff: datetime) -> PredictionEnvelope:
    return PredictionEnvelope(
        prediction_id="prediction_" + str(int(cutoff.timestamp())),
        decision_cutoff_utc=cutoff,
        deployment_pseudonym=DEPLOYMENT,
        snapshot_mode=SnapshotMode.PROSPECTIVE,
        telemetry_snapshot_hash="a" * 64,
        topology_snapshot_hash="b" * 64,
        operational_profile_hash="c" * 64,
        feature_schema_hash="d" * 64,
        feature_vector_hash="e" * 64,
        model_bundle_hash="f" * 64,
        policy_hash="1" * 64,
        stage1_probability=0.99,
        healthy_residual=5.0,
        support_score=0.1,
        support_axes={"domain": 0.1},
        impact_score=0.9,
        impact_families=("service_availability_decline",),
        causal_consistency=True,
        fast_path_state=DecisionState.INCIDENT,
        slow_path_state=DecisionState.INCIDENT,
        dual_path_state=DecisionState.INCIDENT,
        slow_path_statistic=12.0,
        episode_state_before=DecisionState.HEALTHY,
        episode_state_after=DecisionState.INCIDENT,
        final_decision=DecisionState.INCIDENT,
        activation_path="both",
        stage2_top_k=(
            RankedCause(
                component_pseudonym=ROOT,
                score=0.9,
                evidence={"anomaly": 1.0},
            ),
        ),
        counterfactual_policy_states={
            "F": DecisionState.INCIDENT,
            "S": DecisionState.INCIDENT,
            "D": DecisionState.INCIDENT,
        },
        data_quality_state=DataQualityState.ACCEPTED,
        inference_latency_ms=10.0,
        infrastructure_version="0.7.0",
        created_at_utc=cutoff + timedelta(minutes=1),
    )


def test_blinded_review_reveal_and_disagreement_are_append_only(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    service = AdjudicationService(store)
    service.create_case(
        case_id="case-1",
        deployment_pseudonym=DEPLOYMENT,
        review_window_start_utc=START,
        review_window_end_utc=START + timedelta(hours=4),
        prediction_ids=(),
        candidate_source="existing_monitor",
        severity_hint=IncidentSeverity.HIGH,
        actor_pseudonym=RESOLVER,
    )
    with pytest.raises(ReviewWorkflowError, match="before initial"):
        service.reveal_nrim(
            case_id="case-1",
            reviewer_pseudonym=REVIEWER_A,
        )
    service.submit_initial(
        case_id="case-1",
        adjudication=adjudication(
            adjudication_id="adj-a",
            reviewer=REVIEWER_A,
            version=1,
            blinded=True,
        ),
    )
    assert service.reveal_nrim(
        case_id="case-1",
        reviewer_pseudonym=REVIEWER_A,
    ) == ()
    service.submit_post_reveal(
        case_id="case-1",
        adjudication=adjudication(
            adjudication_id="adj-a",
            reviewer=REVIEWER_A,
            version=2,
            blinded=False,
        ),
    )
    service.submit_initial(
        case_id="case-1",
        adjudication=adjudication(
            adjudication_id="adj-b",
            reviewer=REVIEWER_B,
            version=1,
            blinded=True,
        ),
    )
    service.record_resolution(
        case_id="case-1",
        resolver_pseudonym=RESOLVER,
        final_adjudication=adjudication(
            adjudication_id="adj-final",
            reviewer=RESOLVER,
            version=1,
            blinded=False,
        ),
        disagreement_reason="Independent evidence favored storage.",
    )
    assert len(store.list_adjudications()) == 4
    status = service.workflow_status("case-1")
    assert status["double_review_complete"]
    assert status["resolved"]


def test_review_api_requires_authentication_and_blinded_sequence(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    service = AdjudicationService(store)
    service.create_case(
        case_id="case-api",
        deployment_pseudonym=DEPLOYMENT,
        review_window_start_utc=START,
        review_window_end_utc=START + timedelta(hours=4),
        prediction_ids=(),
        candidate_source="random_healthy_sample",
        actor_pseudonym=RESOLVER,
    )
    app = FastAPI()
    app.include_router(
        create_review_router(
            service=service,
            authorizer=TokenAuthorizer(
                {
                    "secret-review-token": ReviewPrincipal(
                        reviewer_pseudonym=REVIEWER_A,
                        role=ReviewerRole.REVIEWER,
                    )
                }
            ),
        )
    )
    client = TestClient(app)
    assert (
        client.get("/nrim-shadow/review/cases/case-api/status").status_code
        == 401
    )
    headers = {"Authorization": "Bearer secret-review-token"}
    assert (
        client.get(
            "/nrim-shadow/review/cases/case-api/status",
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/nrim-shadow/review/cases/case-api/reveal",
            headers=headers,
        ).status_code
        == 409
    )


def test_review_console_is_same_origin_and_discloses_no_case_data(
    tmp_path,
) -> None:
    service = AdjudicationService(
        AppendOnlyEvidenceStore(tmp_path / "console.sqlite")
    )
    app = FastAPI()
    app.include_router(
        create_review_router(
            service=service,
            authorizer=TokenAuthorizer(
                {
                    "review-token": ReviewPrincipal(
                        reviewer_pseudonym=REVIEWER_A,
                        role=ReviewerRole.REVIEWER,
                    )
                }
            ),
        )
    )
    client = TestClient(app)
    response = client.get("/nrim-shadow/review/console")
    assert response.status_code == 200
    assert "NRIM shadow evidence review" in response.text
    assert DEPLOYMENT not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert "connect-src 'self'" in response.headers[
        "content-security-policy"
    ]
    script = client.get("/nrim-shadow/review/console.js")
    assert script.status_code == 200
    assert "localStorage" not in script.text
    assert "sessionStorage" not in script.text
    assert "textContent" in script.text


def test_metrics_report_nonzero_uncertainty_when_no_false_episodes() -> None:
    assert exact_poisson_rate_upper(events=0, exposure=3000.0) > 0.0
    assert cohen_kappa([("yes", "yes"), ("no", "no")]) == 1.0

    predictions = [
        prediction(START + timedelta(hours=offset))
        for offset in (1, 2, 3)
    ]
    labels = [
        adjudication(
            adjudication_id="incident-1",
            reviewer=REVIEWER_A,
            version=1,
            blinded=False,
        )
    ]
    report = ProspectiveMetricsEngine(
        decision_interval_hours=1.0,
        bootstrap_seed=7,
    ).evaluate(
        predictions=predictions,
        adjudications=labels,
        healthy_exposures=(
            HealthyExposure(
                deployment_pseudonym=DEPLOYMENT,
                topology_family="family-a",
                healthy_hours=3000.0,
                nonactionable_episodes={"F": 0, "S": 0, "D": 0},
            ),
        ),
        agreement_pairs_impact=(("yes", "yes"),),
        agreement_pairs_cause=(("storage", "storage"),),
    )
    assert report["incident_detection"]["D"]["episode_recall"] == 1.0
    assert report["ranking"]["mrr"] == 1.0
    assert (
        report["alert_burden"]["D"]["conservative_upper_95"]
        > 0.0
    )


def test_operational_recall_does_not_drop_unsupported_incidents() -> None:
    detected = IncidentCase(
        incident_id="supported",
        deployment_pseudonym=DEPLOYMENT,
        failure_family="storage",
        severity=IncidentSeverity.HIGH,
        onset_utc=START,
        recovery_utc=START + timedelta(hours=1),
        in_support=True,
        data_quality_blocked=False,
        detections={"F": START, "S": START, "D": START},
        fragments={"F": 0, "S": 0, "D": 0},
        confirmed_root_component=ROOT,
        stage2_top_k=(ROOT,),
        ranking_available=True,
        engineer_top3_useful=None,
    )
    unsupported = IncidentCase(
        incident_id="unsupported",
        deployment_pseudonym=DEPLOYMENT,
        failure_family="storage",
        severity=IncidentSeverity.HIGH,
        onset_utc=START,
        recovery_utc=START + timedelta(hours=1),
        in_support=False,
        data_quality_blocked=True,
        detections={"F": None, "S": None, "D": None},
        fragments={"F": 0, "S": 0, "D": 0},
        confirmed_root_component=ROOT,
        stage2_top_k=(),
        ranking_available=False,
        engineer_top3_useful=None,
    )
    result = ProspectiveMetricsEngine(
        decision_interval_hours=1.0
    )._detection((detected, unsupported), "D")
    assert result["included_incident_count"] == 2
    assert result["in_support_incident_count"] == 1
    assert result["episode_recall"] == 0.5
    assert result["conditional_in_support_recall"] == 1.0


def test_synthetic_results_can_never_promote_release() -> None:
    report = {
        "incident_detection": {
            "D": {
                "episode_recall": 1.0,
                "severe_incident_count": 30,
                "severe_recall": 1.0,
                "failure_families": {
                    "storage": {"incident_count": 30, "recall": 1.0}
                },
                "median_delay_hours": 0.0,
                "p90_delay_hours": 0.0,
                "mean_extra_fragments": 0.0,
            }
        },
        "alert_burden": {
            "D": {
                "episodes_per_100_healthy_hours": 0.0,
                "conservative_upper_95": 0.1,
                "maximum_deployment_concentration_ratio": 0.0,
            }
        },
        "ranking": {
            "mrr": 1.0,
            "hits_at_1": 1.0,
            "hits_at_3": 1.0,
            "ranking_coverage": 1.0,
        },
        "support_and_data_quality": {
            "unsupported_safe_semantics": True,
            "unsafe_stage2_count": 0,
        },
        "path_attribution": {"recommendation": "retain_slow_only"},
        "adjudication": {
            "impact_kappa": 1.0,
            "cause_family_kappa": 1.0,
        },
        "human_utility": {},
    }
    operational = {
        "schema_compliance": 1.0,
        "feature_parity_fraction": 1.0,
        "future_leakage_count": 0,
        "prediction_reproducibility_fraction": 1.0,
        "topology_provenance_fraction": 1.0,
        "audit_field_leakage_count": 0,
        "scheduled_cutoff_completion_fraction": 1.0,
        "evidence_durability_fraction": 1.0,
        "p95_latency_seconds": 1.0,
        "p95_latency_fraction_of_decision_interval": 0.01,
        "unregistered_bundle_change_count": 0,
        "state_recovery_parity_fraction": 1.0,
        "decision_interval_hours": 1.0,
    }
    decision = evaluate_release(
        report=report,
        operational=operational,
        context=PilotContext(
            evidence_origin=EvidenceOrigin.SYNTHETIC,
            pilot_start_utc=START,
            pilot_end_utc=START + timedelta(weeks=10),
            independent_deployment_count=5,
            topology_family_count=5,
            healthy_deployment_hours=3000,
            adjudicated_incident_count=30,
            protocol_registered_before_outcomes=True,
        ),
    )
    assert decision.state is PromotionState.CANDIDATE
    assert "real_prospective_evidence" in decision.blockers


def test_protocol_is_signed_preregistered_and_cannot_be_overwritten(
    tmp_path,
) -> None:
    private, public = generate_signing_keypair()
    public_path = tmp_path / "protocol_public.pem"
    public_path.write_bytes(public)
    protocol = default_protocol(
        planned_start_utc=START + timedelta(days=1),
        planned_end_utc=START + timedelta(weeks=9),
        deployment_pseudonyms=tuple(
            f"deployment_{index}_" + "a" * 32
            for index in range(5)
        ),
        topology_family_count=5,
        model_bundle_hash="a" * 64,
        bundle_public_key_sha256=bytes_hash(public),
        feature_schema_hash="b" * 64,
        policy_hash="c" * 64,
    )
    path = tmp_path / "protocol_registration.json"
    protocol_hash = register_protocol(
        protocol=protocol,
        registration_path=path,
        private_key=private,
        public_key_pem=public,
    )
    assert verify_registered_protocol(
        registration_path=path,
        public_key_path=public_path,
    ) == protocol
    assert len(protocol_hash) == 64
    with pytest.raises(FileExistsError):
        register_protocol(
            protocol=protocol,
            registration_path=path,
            private_key=private,
            public_key_pem=public,
        )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["protocol"]["minimum_adjudicated_incidents"] = 1
    bad_path = tmp_path / "tampered_protocol.json"
    bad_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment"):
        verify_registered_protocol(
            registration_path=bad_path,
            public_key_path=public_path,
        )
