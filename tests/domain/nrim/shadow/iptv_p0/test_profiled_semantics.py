from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.batch_adapter import (
    BatchInputRecord,
    validate_profile_semantics,
)
from app.domain.nrim.shadow.iptv_p0.signal_registry import (
    AvailabilityClass,
    SignalRegistry,
    SignalRequirement,
    SupportConsequence,
)


def _record(record_factory, **updates):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="semantic-record",
    )
    value = record.model_dump(mode="json")
    value.update(updates)
    return BatchInputRecord.model_validate(value)


@pytest.mark.parametrize(
    "updates",
    (
        {"unit": "milliseconds"},
        {"metric_value": -1.0},
        {"metric_value": 101.0},
        {"applicability": "unsupported"},
        {"source_id": "src_" + "f" * 32},
    ),
)
def test_registered_semantic_mutations_are_rejected(
    record_factory,
    registry,
    updates,
):
    if updates.get("applicability") == "unsupported":
        with pytest.raises(ValidationError):
            _record(record_factory, **updates)
        return
    record = _record(record_factory, **updates)
    with pytest.raises(ValueError):
        validate_profile_semantics(record, registry=registry)


def test_registered_positive_fixture_is_accepted(record_factory, registry):
    validate_profile_semantics(_record(record_factory), registry=registry)


def test_unavailable_signal_observation_is_rejected(
    record_factory,
    registry,
):
    signal = "system.disk.utilization"
    requirements = []
    for item in registry.requirements:
        if item.signal_name == signal:
            value = item.model_dump(mode="json")
            value.update(
                {
                    "availability": AvailabilityClass.UNAVAILABLE,
                    "support_consequence": SupportConsequence.ESCALATE,
                    "source_ids": (),
                }
            )
            item = SignalRequirement.model_validate(value)
        requirements.append(item)
    unavailable = SignalRegistry(
        registry_id=registry.registry_id,
        registry_version=registry.registry_version,
        requirements=tuple(requirements),
    )
    with pytest.raises(ValueError, match="Unavailable"):
        validate_profile_semantics(
            _record(record_factory),
            registry=unavailable,
        )


def test_profile_semantics_are_deterministic(record_factory, registry):
    record = _record(record_factory)
    assert (
        validate_profile_semantics(record, registry=registry)
        is validate_profile_semantics(record, registry=registry)
        is None
    )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_source_values_are_contract_invalid(record_factory, value):
    with pytest.raises(ValidationError):
        _record(record_factory, metric_value=value)
