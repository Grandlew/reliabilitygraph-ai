from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.clock_watermark import (
    ClockWatermarkPolicy,
    TimeDisposition,
    assess_record_time,
    deterministic_watermark,
)
from app.domain.nrim.shadow.iptv_p0.source_inventory import ClockQuality

BASE_TIME = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "offset_hours",
    (-8, 0, 5),
)
def test_time_assessment_normalizes_explicit_offsets(
    offset_hours,
    clock_policy,
):
    zone = timezone(timedelta(hours=offset_hours))
    event = BASE_TIME.astimezone(zone)
    result = assess_record_time(
        event_time_utc=event,
        observation_time_utc=(BASE_TIME + timedelta(seconds=1)).astimezone(
            zone
        ),
        ingestion_time_utc=(BASE_TIME + timedelta(seconds=2)).astimezone(zone),
        clock_quality=ClockQuality.VERIFIED_SYNCHRONIZED,
        policy=clock_policy,
    )
    assert result.disposition is TimeDisposition.ACCEPT
    assert result.event_time_utc == BASE_TIME


@pytest.mark.parametrize(
    "field",
    ("event_time_utc", "observation_time_utc", "ingestion_time_utc"),
)
def test_time_assessment_rejects_ambiguous_timezone(field, clock_policy):
    values = {
        "event_time_utc": BASE_TIME,
        "observation_time_utc": BASE_TIME + timedelta(seconds=1),
        "ingestion_time_utc": BASE_TIME + timedelta(seconds=2),
        "clock_quality": ClockQuality.VERIFIED_SYNCHRONIZED,
        "policy": clock_policy,
    }
    values[field] = datetime(2026, 1, 1, 1, 0)
    with pytest.raises((ValidationError, ValueError)):
        assess_record_time(**values)


@pytest.mark.parametrize(
    "clock_quality",
    (ClockQuality.UNKNOWN, ClockQuality.UNSYNCHRONIZED),
)
def test_unknown_clock_semantics_quarantine(clock_quality, clock_policy):
    result = assess_record_time(
        event_time_utc=BASE_TIME,
        observation_time_utc=BASE_TIME + timedelta(seconds=1),
        ingestion_time_utc=BASE_TIME + timedelta(seconds=2),
        clock_quality=clock_quality,
        policy=clock_policy,
    )
    assert result.disposition is TimeDisposition.QUARANTINE
    assert result.reason_code == "UNKNOWN_OR_UNSYNCHRONIZED_CLOCK"


@pytest.mark.parametrize(
    "event_delta,observation_delta,ingestion_delta,reason",
    (
        (10, 11, 0, "FUTURE_EVENT_TIME"),
        (0, -1, 1, "OBSERVATION_PRECEDES_EVENT"),
        (0, 2, 1, "INGESTION_PRECEDES_OBSERVATION"),
        (0, 61, 62, "OBSERVATION_DELAY_EXCEEDED"),
    ),
)
def test_invalid_clock_relationships_quarantine(
    event_delta,
    observation_delta,
    ingestion_delta,
    reason,
    clock_policy,
):
    result = assess_record_time(
        event_time_utc=BASE_TIME + timedelta(seconds=event_delta),
        observation_time_utc=BASE_TIME
        + timedelta(seconds=observation_delta),
        ingestion_time_utc=BASE_TIME + timedelta(seconds=ingestion_delta),
        clock_quality=ClockQuality.BOUNDED_SKEW,
        policy=clock_policy,
    )
    assert result.disposition is TimeDisposition.QUARANTINE
    assert result.reason_code == reason


def test_record_at_watermark_is_late(clock_policy):
    result = assess_record_time(
        event_time_utc=BASE_TIME,
        observation_time_utc=BASE_TIME + timedelta(seconds=1),
        ingestion_time_utc=BASE_TIME + timedelta(seconds=2),
        clock_quality=ClockQuality.BOUNDED_SKEW,
        policy=clock_policy,
        watermark_utc=BASE_TIME,
    )
    assert result.disposition is TimeDisposition.LATE


@pytest.mark.parametrize("reverse", (False, True))
def test_watermark_is_source_order_independent(reverse, clock_policy):
    values = [
        ("src_a", (BASE_TIME, BASE_TIME + timedelta(seconds=30))),
        ("src_b", (BASE_TIME + timedelta(seconds=10),)),
    ]
    if reverse:
        values.reverse()
    result = deterministic_watermark(dict(values), policy=clock_policy)
    assert result == BASE_TIME - timedelta(seconds=110)


def test_watermark_rejects_empty_inventory(clock_policy):
    with pytest.raises(ValueError):
        deterministic_watermark({}, policy=clock_policy)


def test_watermark_rejects_source_without_events(clock_policy):
    with pytest.raises(ValueError):
        deterministic_watermark({"src_a": ()}, policy=clock_policy)


def test_unknown_clock_blocking_cannot_be_disabled(clock_policy):
    value = clock_policy.model_dump(mode="json")
    value["unknown_clock_blocks"] = False
    with pytest.raises(ValidationError):
        ClockWatermarkPolicy.model_validate(value)
