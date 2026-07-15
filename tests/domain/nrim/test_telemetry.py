from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.telemetry import (
    BaselineStatus,
    CanonicalTelemetryEvent,
    TelemetryBatch,
    TelemetryQuality,
    TelemetrySignalType,
)


def make_metric_event(
    *,
    event_id: str = "event_1",
    deployment_id: str = "deployment_1",
    component_node_id: str = "component_1",
) -> CanonicalTelemetryEvent:
    now = datetime.now(timezone.utc)

    return CanonicalTelemetryEvent(
        event_id=event_id,
        deployment_id=deployment_id,
        component_node_id=component_node_id,
        observed_at=now,
        ingested_at=now,
        signal_type=TelemetrySignalType.METRIC,
        signal_name="system.cpu.utilization",
        value=75.0,
        unit="percent",
        collection_source="test_collector",
        quality=TelemetryQuality.HIGH,
    )


def test_numeric_metric_requires_unit() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        CanonicalTelemetryEvent(
            deployment_id="deployment_1",
            component_node_id="component_1",
            observed_at=now,
            ingested_at=now,
            signal_type=TelemetrySignalType.METRIC,
            signal_name="system.cpu.utilization",
            value=75.0,
            unit=None,
            collection_source="test_collector",
            quality=TelemetryQuality.HIGH,
        )


def test_deviation_requires_baseline_value() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        CanonicalTelemetryEvent(
            deployment_id="deployment_1",
            component_node_id="component_1",
            observed_at=now,
            ingested_at=now,
            signal_type=TelemetrySignalType.METRIC,
            signal_name="system.cpu.utilization",
            value=75.0,
            unit="percent",
            collection_source="test_collector",
            quality=TelemetryQuality.HIGH,
            deviation_score=3.2,
        )


def test_quarantined_record_cannot_be_marked_normal() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        CanonicalTelemetryEvent(
            deployment_id="deployment_1",
            component_node_id="component_1",
            observed_at=now,
            ingested_at=now,
            signal_type=TelemetrySignalType.METRIC,
            signal_name="system.cpu.utilization",
            value=75.0,
            unit="percent",
            collection_source="test_collector",
            quality=TelemetryQuality.QUARANTINED,
            baseline_status=BaselineStatus.NORMAL,
        )


def test_batch_rejects_duplicate_event_ids() -> None:
    event_1 = make_metric_event(event_id="duplicate")
    event_2 = make_metric_event(event_id="duplicate")

    with pytest.raises(ValidationError):
        TelemetryBatch(
            deployment_id="deployment_1",
            source="test",
            events=[event_1, event_2],
        )


def test_batch_rejects_mixed_deployments() -> None:
    event_1 = make_metric_event(deployment_id="deployment_1")
    event_2 = make_metric_event(
        event_id="event_2",
        deployment_id="deployment_2",
    )

    with pytest.raises(ValidationError):
        TelemetryBatch(
            deployment_id="deployment_1",
            source="test",
            events=[event_1, event_2],
        )


def test_future_observed_time_is_rejected() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        CanonicalTelemetryEvent(
            deployment_id="deployment_1",
            component_node_id="component_1",
            observed_at=now + timedelta(hours=2),
            ingested_at=now,
            signal_type=TelemetrySignalType.METRIC,
            signal_name="system.cpu.utilization",
            value=75.0,
            unit="percent",
            collection_source="test_collector",
            quality=TelemetryQuality.HIGH,
        )


def test_user_reported_symptom_requires_evidence() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        CanonicalTelemetryEvent(
            deployment_id="deployment_1",
            component_node_id="component_1",
            observed_at=now,
            ingested_at=now,
            signal_type=TelemetrySignalType.USER_REPORTED_SYMPTOM,
            signal_name="iptv.playback.freezing_report",
            value={
                "affected_floor": 3
            },
            collection_source="hotel_staff",
            quality=TelemetryQuality.LOW,
            evidence_ids=[],
        )
