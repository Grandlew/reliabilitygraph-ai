import pytest
from pydantic import ValidationError

from app.domain.nrim.models import (
    DiagnosticTest,
    Evidence,
    NRIMEdge,
    NRIMNode,
    OperationalObservation,
    ReliabilityCase,
)
from app.domain.nrim.ontology import (
    EdgeType,
    EvidenceKind,
    KnowledgeStatus,
    NodeCategory,
    ProductMode,
    SignalType,
)


def make_evidence() -> Evidence:
    return Evidence(
        kind=EvidenceKind.DETERMINISTIC_RULE,
        statement="Supported by a deterministic test rule.",
        source_reference="TEST_RULE",
        confidence=1.0,
    )


def make_architecture_node(node_id: str) -> NRIMNode:
    return NRIMNode(
        id=node_id,
        category=NodeCategory.ARCHITECTURE,
        type="Middleware",
        name="Middleware",
        status=KnowledgeStatus.CONFIRMED,
        evidence=[],
    )


def make_failure_node(node_id: str) -> NRIMNode:
    return NRIMNode(
        id=node_id,
        category=NodeCategory.FAILURE,
        type="RootCauseHypothesis",
        name="Possible middleware failure",
        status=KnowledgeStatus.INFERRED,
        evidence=[make_evidence()],
    )


def test_inferred_node_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NRIMNode(
            id="node_1",
            category=NodeCategory.ARCHITECTURE,
            type="Storage",
            name="Storage",
            status=KnowledgeStatus.INFERRED,
            evidence=[],
        )


def test_inferred_edge_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NRIMEdge(
            source_node_id="node_1",
            target_node_id="node_2",
            type=EdgeType.DEPENDS_ON,
            status=KnowledgeStatus.INFERRED,
            evidence=[],
        )


def test_duplicate_node_ids_are_rejected() -> None:
    node_1 = make_architecture_node("duplicate")
    node_2 = make_architecture_node("duplicate")

    with pytest.raises(ValidationError):
        ReliabilityCase(
            project_id="project_1",
            mode=ProductMode.ARCHITECTURE_ASSURANCE,
            nodes=[node_1, node_2],
        )


def test_edge_with_missing_target_is_rejected() -> None:
    node = make_architecture_node("node_1")

    edge = NRIMEdge(
        source_node_id="node_1",
        target_node_id="missing_node",
        type=EdgeType.DEPENDS_ON,
        status=KnowledgeStatus.INFERRED,
        evidence=[make_evidence()],
    )

    with pytest.raises(ValidationError):
        ReliabilityCase(
            project_id="project_1",
            mode=ProductMode.ARCHITECTURE_ASSURANCE,
            nodes=[node],
            edges=[edge],
        )


def test_self_loop_is_rejected() -> None:
    node = make_architecture_node("node_1")

    edge = NRIMEdge(
        source_node_id="node_1",
        target_node_id="node_1",
        type=EdgeType.DEPENDS_ON,
        status=KnowledgeStatus.INFERRED,
        evidence=[make_evidence()],
    )

    with pytest.raises(ValidationError):
        ReliabilityCase(
            project_id="project_1",
            mode=ProductMode.ARCHITECTURE_ASSURANCE,
            nodes=[node],
            edges=[edge],
        )


def test_diagnostic_test_must_reference_failure_node() -> None:
    architecture_node = make_architecture_node("middleware_1")

    diagnostic_test = DiagnosticTest(
        name="Check middleware health",
        question="Is the middleware responding normally?",
        hypothesis_ids=["middleware_1"],
        estimated_minutes=5,
        cost_level=1,
        invasiveness_level=1,
        operational_risk_level=1,
        expected_information_gain=0.7,
    )

    with pytest.raises(ValidationError):
        ReliabilityCase(
            project_id="project_1",
            mode=ProductMode.ROOT_CAUSE_DIAGNOSIS,
            nodes=[architecture_node],
            diagnostic_tests=[diagnostic_test],
        )


def test_observation_must_reference_existing_component() -> None:
    observation = OperationalObservation(
        deployment_id="deployment_1",
        component_node_id="missing_component",
        timestamp="2026-07-12T12:00:00Z",
        signal_type=SignalType.METRIC,
        signal_name="cpu_utilization",
        value=90.0,
        unit="percent",
        collection_source="test",
    )

    with pytest.raises(ValidationError):
        ReliabilityCase(
            project_id="project_1",
            deployment_id="deployment_1",
            mode=ProductMode.OPERATIONAL_HEALTH,
            observations=[observation],
        )
