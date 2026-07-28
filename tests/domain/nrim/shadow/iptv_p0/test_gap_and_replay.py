from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.contracts import LabelKind
from app.domain.nrim.shadow.iptv_p0.gap_analysis import (
    CoverageArea,
    CoverageMeasurement,
    SafeRoute,
    compile_gap_report,
)
from app.domain.nrim.shadow.iptv_p0.replay_protocol import (
    CohortDefinition,
    HistoricalReplayProtocol,
    ReplayMetric,
    replay_readiness,
)


@pytest.mark.parametrize("area", tuple(CoverageArea))
def test_each_coverage_area_emits_reason_coded_gap(area):
    report = compile_gap_report(
        (
            CoverageMeasurement(
                area=area,
                subject_id="subject",
                expected_count=10,
                qualified_count=8,
                minimum_fraction=0.9,
                failure_route=SafeRoute.BLOCKED,
            ),
        )
    )
    assert report.gaps[0].area is area
    assert report.gaps[0].route is SafeRoute.BLOCKED
    assert report.gaps[0].reason_code.endswith("COVERAGE_BELOW_MINIMUM")


def test_complete_coverage_has_no_gap():
    report = compile_gap_report(
        (
            CoverageMeasurement(
                area=CoverageArea.SOURCE,
                subject_id="subject",
                expected_count=10,
                qualified_count=10,
                minimum_fraction=1.0,
                failure_route=SafeRoute.BLOCKED,
            ),
        )
    )
    assert report.passed
    assert report.terminal_route is None


def test_gap_report_uses_safest_most_severe_route():
    measurements = tuple(
        CoverageMeasurement(
            area=area,
            subject_id=area.value,
            expected_count=1,
            qualified_count=0,
            minimum_fraction=1.0,
            failure_route=route,
        )
        for area, route in (
            (CoverageArea.SOURCE, SafeRoute.UNKNOWN),
            (CoverageArea.CLOCK, SafeRoute.ESCALATE),
            (CoverageArea.FEATURE, SafeRoute.BLOCKED),
        )
    )
    assert compile_gap_report(measurements).terminal_route is SafeRoute.BLOCKED


def test_gap_report_rejects_duplicate_measurement_identity():
    measurement = CoverageMeasurement(
        area=CoverageArea.SOURCE,
        subject_id="subject",
        expected_count=1,
        qualified_count=0,
        minimum_fraction=1.0,
        failure_route=SafeRoute.UNKNOWN,
    )
    with pytest.raises(ValueError, match="unique"):
        compile_gap_report((measurement, measurement))


@pytest.mark.parametrize(
    "field",
    (
        "no_model_tuning",
        "no_threshold_tuning",
        "no_feature_tuning",
        "no_support_rule_tuning",
        "no_watermark_tuning",
        "no_episode_grouping_tuning",
    ),
)
def test_replay_protocol_rejects_every_tuning_authority(
    replay_protocol,
    field,
):
    value = replay_protocol.model_dump(mode="json")
    value[field] = False
    with pytest.raises(ValidationError):
        HistoricalReplayProtocol.model_validate(value)


@pytest.mark.parametrize("missing", tuple(LabelKind))
def test_replay_protocol_requires_separate_outcome_labels(
    replay_protocol,
    missing,
):
    value = replay_protocol.model_dump(mode="json")
    value["labels"] = [
        item for item in value["labels"] if item["label_kind"] != missing.value
    ]
    with pytest.raises(ValidationError):
        HistoricalReplayProtocol.model_validate(value)


@pytest.mark.parametrize("missing", tuple(ReplayMetric))
def test_replay_protocol_requires_all_preregistered_metrics(
    replay_protocol,
    missing,
):
    value = replay_protocol.model_dump(mode="json")
    value["metrics"] = [
        item for item in value["metrics"] if item != missing.value
    ]
    with pytest.raises(ValidationError):
        HistoricalReplayProtocol.model_validate(value)


def test_replay_cohort_requires_positive_event_time_window(replay_protocol):
    cohort = replay_protocol.cohorts[0]
    with pytest.raises(ValidationError):
        CohortDefinition(
            cohort_id=cohort.cohort_id,
            inclusion_rule=cohort.inclusion_rule,
            exclusion_rule=cohort.exclusion_rule,
            event_time_start_utc=cohort.event_time_end_utc,
            event_time_end_utc=cohort.event_time_end_utc,
        )


@pytest.mark.parametrize(
    "real,criteria,decision",
    (
        (False, False, "REAL_DATA_REQUIRED"),
        (True, False, "BLOCKED"),
        (True, True, "REAL_REPLAY_READY"),
    ),
)
def test_replay_readiness_state_machine(
    replay_protocol,
    real,
    criteria,
    decision,
):
    result = replay_readiness(
        protocol=replay_protocol,
        real_deployment_pack_qualified=real,
        qualification_criteria_passed=criteria,
    )
    assert result["decision"] == decision
    assert result["tuning_authorized"] is False


def test_replay_protocol_hash_is_deterministic(replay_protocol):
    assert replay_protocol.content_hash() == replay_protocol.content_hash()
