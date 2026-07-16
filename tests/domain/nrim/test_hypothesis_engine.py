from datetime import datetime, timezone

from app.domain.nrim.hypothesis_engine import (
    build_evidence_links,
    calculate_link_weight,
    create_catchup_storage_hypotheses,
    mark_leading_hypothesis,
    score_hypothesis,
)
from app.domain.nrim.reasoning import (
    EvidenceRole,
    HypothesisStatus,
)
from app.domain.nrim.rule_catalog import (
    CATCHUP_STORAGE_RULES,
)
from app.domain.nrim.telemetry import (
    CanonicalTelemetryEvent,
    GoldenSignal,
    TelemetryQuality,
    TelemetrySignalType,
)


def make_metric(
    *,
    event_id: str,
    signal_name: str,
    value: float,
    unit: str,
    quality: TelemetryQuality = TelemetryQuality.HIGH,
) -> CanonicalTelemetryEvent:
    now = datetime.now(timezone.utc)

    return CanonicalTelemetryEvent(
        event_id=event_id,
        deployment_id="deployment_1",
        component_node_id="arch_catchup_storage",
        service_node_ids=["service_catchup"],
        observed_at=now,
        ingested_at=now,
        signal_type=TelemetrySignalType.METRIC,
        signal_name=signal_name,
        value=value,
        unit=unit,
        collection_source="test_source",
        quality=quality,
        golden_signal=GoldenSignal.SATURATION,
    )


def test_high_storage_utilization_supports_capacity_hypothesis() -> None:
    event = make_metric(
        event_id="event_1",
        signal_name="system.disk.utilization",
        value=90.0,
        unit="percent",
    )

    hypothesis = create_catchup_storage_hypotheses()[0]

    links = build_evidence_links(
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_type=hypothesis.failure_type,
        events=[event],
        rules=CATCHUP_STORAGE_RULES,
    )

    assert links
    assert any(
        link.evidence.role == EvidenceRole.SUPPORTS
        for link in links
    )


def test_quarantined_evidence_contributes_zero_weight() -> None:
    event = make_metric(
        event_id="event_1",
        signal_name="system.disk.utilization",
        value=96.0,
        unit="percent",
        quality=TelemetryQuality.QUARANTINED,
    )

    hypothesis = create_catchup_storage_hypotheses()[0]

    links = build_evidence_links(
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_type=hypothesis.failure_type,
        events=[event],
        rules=CATCHUP_STORAGE_RULES,
    )

    assert links
    assert all((link.weight or 0.0) == 0.0 for link in links)


def test_highest_scoring_hypothesis_becomes_leading() -> None:
    hypotheses = create_catchup_storage_hypotheses()

    updated = []

    for index, hypothesis in enumerate(hypotheses):
        updated.append(
            hypothesis.model_copy(
                update={
                    "ranking_score": float(index),
                    "status": HypothesisStatus.SUPPORTED,
                }
            )
        )

    ranked = mark_leading_hypothesis(updated)

    assert ranked[0].status == HypothesisStatus.LEADING
