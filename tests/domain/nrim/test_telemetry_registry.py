import pytest
from pydantic import ValidationError

from app.domain.nrim.telemetry_registry import (
    ComponentMapping,
    ComponentRegistry,
    ExternalIdentity,
)


def make_identity() -> ExternalIdentity:
    return ExternalIdentity(
        source_type="hostname",
        source_system="linux_agent",
        external_identifier="middleware-01",
    )


def test_registry_rejects_ambiguous_active_mapping() -> None:
    identity = make_identity()

    mapping_1 = ComponentMapping(
        deployment_id="deployment_1",
        component_node_id="middleware_primary",
        identities=[identity],
        mapping_confidence=1.0,
    )

    mapping_2 = ComponentMapping(
        deployment_id="deployment_1",
        component_node_id="middleware_secondary",
        identities=[identity],
        mapping_confidence=1.0,
    )

    with pytest.raises(ValidationError):
        ComponentRegistry(
            mappings=[mapping_1, mapping_2]
        )


def test_registry_resolves_component() -> None:
    identity = make_identity()

    registry = ComponentRegistry(
        mappings=[
            ComponentMapping(
                deployment_id="deployment_1",
                component_node_id="middleware_primary",
                identities=[identity],
                mapping_confidence=1.0,
            )
        ]
    )

    result = registry.resolve(
        deployment_id="deployment_1",
        source_type="hostname",
        source_system="linux_agent",
        external_identifier="middleware-01",
    )

    assert result == "middleware_primary"


def test_registry_returns_none_when_mapping_missing() -> None:
    registry = ComponentRegistry(mappings=[])

    result = registry.resolve(
        deployment_id="deployment_1",
        source_type="hostname",
        source_system="linux_agent",
        external_identifier="unknown-host",
    )

    assert result is None
