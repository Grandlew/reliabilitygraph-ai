from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ExternalIdentity(BaseModel):
    source_type: str = Field(min_length=1)
    source_system: str = Field(min_length=1)
    external_identifier: str = Field(min_length=1)

    def identity_key(self) -> tuple[str, str, str]:
        return (
            self.source_type,
            self.source_system,
            self.external_identifier,
        )


class ComponentMapping(BaseModel):
    deployment_id: str = Field(min_length=1)
    component_node_id: str = Field(min_length=1)
    identities: list[ExternalIdentity] = Field(min_length=1)
    active: bool = True
    mapping_confidence: float = Field(ge=0.0, le=1.0)
    verified_by: str | None = None


class ComponentRegistry(BaseModel):
    mappings: list[ComponentMapping] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_registry(self) -> "ComponentRegistry":
        seen_identities: dict[tuple[str, tuple[str, str, str]], str] = {}

        for mapping in self.mappings:
            if not mapping.active:
                continue

            for identity in mapping.identities:
                key = (mapping.deployment_id, identity.identity_key())
                if key in seen_identities and seen_identities[key] != mapping.component_node_id:
                    raise ValueError(
                        f"Identity {identity.identity_key()} is mapped to multiple active components "
                        f"({seen_identities[key]}, {mapping.component_node_id}) in deployment {mapping.deployment_id}"
                    )
                seen_identities[key] = mapping.component_node_id

        return self

    def resolve(
        self,
        *,
        deployment_id: str,
        source_type: str,
        source_system: str,
        external_identifier: str,
    ) -> str | None:
        target_key = (source_type, source_system, external_identifier)
        matches = [
            m.component_node_id for m in self.mappings
            if m.active and m.deployment_id == deployment_id
            and any(ident.identity_key() == target_key for ident in m.identities)
        ]

        if not matches:
            return None

        unique_matches = list(set(matches))
        if len(unique_matches) > 1:
            raise ValueError(
                f"Multiple active mappings found for identity {target_key} in deployment {deployment_id}")

        return unique_matches[0]
