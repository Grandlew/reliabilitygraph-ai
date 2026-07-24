from __future__ import annotations

from app.domain.nrim.shadow.monitoring import (
    DataQualityMonitor,
    FeatureDriftMonitor,
    RuntimeHealthMonitor,
    structured_runtime_event,
)


def test_runtime_and_data_quality_monitoring_contracts() -> None:
    runtime = RuntimeHealthMonitor(decision_interval_seconds=60.0)
    runtime.record_cutoff(
        envelope=None,
        evidence_write_succeeded=False,
        backlog_depth=3,
    )
    assert runtime.summary()["evidence_durability_fraction"] == 0.0

    quality = DataQualityMonitor()
    quality.record_contract(accepted=True)
    quality.record_contract(
        accepted=False,
        blocking_category_failure=True,
    )
    quality.record_parity(matches=True)
    quality.record_prediction_audit(
        topology_linked=True,
        reproducible=True,
    )
    summary = quality.summary()
    assert summary["schema_compliance"] == 0.5
    assert summary["feature_parity_fraction"] == 1.0


def test_feature_drift_is_monitoring_only_and_explicit() -> None:
    monitor = FeatureDriftMonitor(
        registered_ranges={"x": (0.0, 1.0)},
        registered_medians={"x": 0.5},
        minimum_batch=2,
        median_shift_fraction=0.25,
    )
    result = monitor.evaluate([{"x": 2.0}, {"x": 2.1}])
    assert result["status"] == "investigate"
    assert result["features"]["x"]["out_of_range_fraction"] == 1.0

    event = structured_runtime_event(
        event_name="nrim.shadow.bundle_mismatch",
        severity="error",
        attributes={"bundle_hash": "abc"},
    )
    assert event["event.name"] == "nrim.shadow.bundle_mismatch"
    assert "nrim.shadow.bundle_hash" in event["attributes"]
