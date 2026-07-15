from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.baseline import (
    BaselineWindow,
    assess_against_baseline,
)
from app.domain.nrim.telemetry import (
    BaselineStatus,
    CanonicalTelemetryEvent,
    TelemetryQuality,
    TelemetrySignalType,
)


def make_event(value: float) -> CanonicalTelemetryEvent:
    now = datetime.now(timezone.utc)

    return CanonicalTelemetryEvent(
        deployment_id="deployment_1",
        component_node_id="storage_1",
        observed_at=now,
        ingested_at=now,
        signal_type=TelemetrySignalType.METRIC,
        signal_name="system.disk.utilization",
        value=value,
        unit="percent",
        collection_source="test",
        quality=TelemetryQuality.HIGH,
    )


def make_baseline(sample_count: int = 100) -> BaselineWindow:
    now = datetime.now(timezone.utc)

    return BaselineWindow(
        deployment_id="deployment_1",
        component_node_id="storage_1",
        signal_name="system.disk.utilization",
        sample_count=sample_count,
        mean=60.0,
        standard_deviation=5.0,
        median=60.0,
        p95=68.0,
        minimum=45.0,
        maximum=72.0,
        window_start=now - timedelta(days=14),
        window_end=now,
    )


def test_invalid_window_is_rejected() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        BaselineWindow(
            deployment_id="deployment_1",
            component_node_id="storage_1",
            signal_name="system.disk.utilization",
            sample_count=0,
            window_start=now,
            window_end=now - timedelta(hours=1),
        )


def test_zero_samples_cannot_have_statistics() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        BaselineWindow(
            deployment_id="deployment_1",
            component_node_id="storage_1",
            signal_name="system.disk.utilization",
            sample_count=0,
            mean=50.0,
            window_start=now - timedelta(days=1),
            window_end=now,
        )


def test_insufficient_history() -> None:
    result = assess_against_baseline(
        make_event(80.0),
        make_baseline(sample_count=10),
    )

    assert result.status == BaselineStatus.INSUFFICIENT_HISTORY


def test_normal_value() -> None:
    result = assess_against_baseline(
        make_event(62.0),
        make_baseline(),
    )

    assert result.status == BaselineStatus.NORMAL


def test_warning_value() -> None:
    result = assess_against_baseline(
        make_event(71.0),
        make_baseline(),
    )

    assert result.status == BaselineStatus.WARNING


def test_anomalous_value() -> None:
    result = assess_against_baseline(
        make_event(78.0),
        make_baseline(),
    )

    assert result.status == BaselineStatus.ANOMALOUS


def test_critical_value() -> None:
    result = assess_against_baseline(
        make_event(90.0),
        make_baseline(),
    )

    assert result.status == BaselineStatus.CRITICAL
