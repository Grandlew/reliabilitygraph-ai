from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.nrim.simulation.feature_schema import SIGNAL_NAMES
from app.domain.nrim.shadow.contracts import EXPECTED_UNIT, MetricName
from app.domain.nrim.shadow.iptv_p0.signal_registry import (
    AvailabilityClass,
    SignalRegistry,
    SignalRequirement,
    SupportConsequence,
)


def _registry(registry, requirements):
    return SignalRegistry(
        registry_id=registry.registry_id,
        registry_version=registry.registry_version,
        requirements=tuple(requirements),
    )


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_every_frozen_signal_has_exactly_one_classification(
    registry,
    signal_name,
):
    requirement = registry.requirement(signal_name)
    assert requirement.signal_name == signal_name
    assert requirement.availability is AvailabilityClass.REQUIRED


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_registry_rejects_each_missing_frozen_signal(registry, signal_name):
    with pytest.raises(ValidationError):
        _registry(
            registry,
            [
                item
                for item in registry.requirements
                if item.signal_name != signal_name
            ],
        )


def test_registry_rejects_duplicate_frozen_signal(registry):
    with pytest.raises(ValidationError):
        _registry(registry, (*registry.requirements, registry.requirements[0]))


def test_registry_rejects_extra_signal(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value["signal_name"] = "unregistered.metric"
    value["canonical_unit"] = "percent"
    with pytest.raises((ValidationError, ValueError)):
        extra = SignalRequirement.model_validate(value)
        _registry(registry, (*registry.requirements[1:], extra))


def test_required_signal_must_block_when_missing(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value["support_consequence"] = SupportConsequence.UNKNOWN
    with pytest.raises(ValidationError):
        SignalRequirement.model_validate(value)


@pytest.mark.parametrize("predicate", (None, "", "  "))
def test_conditional_signal_requires_meaningful_predicate(
    registry,
    predicate,
):
    value = registry.requirements[0].model_dump(mode="json")
    value.update(
        {
            "availability": AvailabilityClass.CONDITIONAL,
            "applicability_predicate": predicate,
        }
    )
    with pytest.raises(ValidationError):
        SignalRequirement.model_validate(value)


def test_nonconditional_signal_rejects_predicate(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value["applicability_predicate"] = "catchup_is_present"
    with pytest.raises(ValidationError):
        SignalRequirement.model_validate(value)


def test_unavailable_signal_cannot_name_a_source(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value.update(
        {
            "availability": AvailabilityClass.UNAVAILABLE,
            "support_consequence": SupportConsequence.ESCALATE,
        }
    )
    with pytest.raises(ValidationError):
        SignalRequirement.model_validate(value)


def test_signal_unit_must_equal_frozen_contract(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value["canonical_unit"] = "seconds"
    with pytest.raises(ValidationError):
        SignalRequirement.model_validate(value)


def test_conditional_false_absence_has_no_support_consequence(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value.update(
        {
            "availability": AvailabilityClass.CONDITIONAL,
            "applicability_predicate": "catchup_is_present",
        }
    )
    conditional = SignalRequirement.model_validate(value)
    modified = _registry(
        registry,
        (conditional, *registry.requirements[1:]),
    )
    assert (
        modified.route_absence(
            conditional.signal_name,
            predicate_value=False,
        )
        is None
    )


def test_conditional_unknown_predicate_blocks(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value.update(
        {
            "availability": AvailabilityClass.CONDITIONAL,
            "applicability_predicate": "catchup_is_present",
        }
    )
    conditional = SignalRequirement.model_validate(value)
    modified = _registry(
        registry,
        (conditional, *registry.requirements[1:]),
    )
    assert (
        modified.route_absence(
            conditional.signal_name,
            predicate_value=None,
        )
        is SupportConsequence.BLOCKED
    )


def test_optional_absence_is_recorded_without_blocking(registry):
    value = registry.requirements[0].model_dump(mode="json")
    value.update(
        {
            "availability": AvailabilityClass.OPTIONAL,
            "support_consequence": SupportConsequence.UNKNOWN,
            "source_ids": (),
        }
    )
    optional = SignalRequirement.model_validate(value)
    modified = _registry(registry, (optional, *registry.requirements[1:]))
    assert modified.route_absence(optional.signal_name) is None


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_registered_units_are_frozen(signal_name, registry):
    assert registry.requirement(signal_name).canonical_unit == (
        EXPECTED_UNIT[MetricName(signal_name)].value
    )


def test_registry_hash_is_order_sensitive_to_semantic_identity(registry):
    first = registry.content_hash()
    reversed_registry = _registry(registry, reversed(registry.requirements))
    assert first != reversed_registry.content_hash()
