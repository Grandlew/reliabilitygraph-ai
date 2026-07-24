from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    ComponentType,
    DependencyType,
    DirectionSemantics,
    TopologyComponent,
    TopologyEdge,
)
from .hashing import canonical_hash
from .replay import TopologyRegistry


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EdgeSemanticRule(StrictModel):
    dependency_type: DependencyType
    direction_semantics: DirectionSemantics
    allowed_source_types: tuple[ComponentType, ...] = Field(min_length=1)
    allowed_destination_types: tuple[ComponentType, ...] = Field(
        min_length=1
    )
    reversal_allowed: bool = False
    rationale: str = Field(min_length=10, max_length=1000)


class TopologySemanticContract(StrictModel):
    contract_id: str = Field(min_length=1, max_length=128)
    contract_version: str = Field(min_length=1, max_length=128)
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    inventory_source: str = Field(min_length=3, max_length=256)
    endpoint_role_provenance: str = Field(min_length=10, max_length=1000)
    validity_interval_provenance: str = Field(min_length=10, max_length=1000)
    edge_rules: tuple[EdgeSemanticRule, ...] = Field(min_length=1)
    registered_change_cutoffs_utc: tuple[datetime, ...] = ()

    @model_validator(mode="after")
    def validate_rules(self):
        keys = [
            (item.dependency_type, item.direction_semantics)
            for item in self.edge_rules
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Topology edge semantic rules must be unique")
        if tuple(sorted(self.registered_change_cutoffs_utc)) != (
            self.registered_change_cutoffs_utc
        ):
            raise ValueError("Registered topology changes must be ordered")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class TopologySemanticError(ValueError):
    pass


@dataclass(frozen=True)
class TopologyAcceptanceReport:
    contract_hash: str
    cutoff_count: int
    reconstruction_fraction: float
    mutation_rejection_fraction: float
    snapshot_hashes: tuple[str, ...]
    change_epoch_count: int
    results: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return (
            self.cutoff_count > 0
            and self.reconstruction_fraction == 1.0
            and self.mutation_rejection_fraction == 1.0
        )


class TopologyAcceptanceHarness:
    def __init__(
        self,
        *,
        contract: TopologySemanticContract,
        registry: TopologyRegistry,
    ) -> None:
        self.contract = contract
        self.registry = registry
        self._rules = {
            (item.dependency_type, item.direction_semantics): item
            for item in contract.edge_rules
        }

    def _validate(
        self,
        *,
        components: tuple[TopologyComponent, ...],
        edges: tuple[TopologyEdge, ...],
    ) -> None:
        component_types = {
            item.component_pseudonym: item.component_type
            for item in components
        }
        for edge in edges:
            rule = self._rules.get(
                (edge.dependency_type, edge.direction_semantics)
            )
            if rule is None:
                raise TopologySemanticError(
                    "Unregistered edge type/direction semantics"
                )
            source_type = component_types[edge.source_component]
            destination_type = component_types[
                edge.destination_component
            ]
            if (
                source_type not in rule.allowed_source_types
                or destination_type
                not in rule.allowed_destination_types
            ):
                raise TopologySemanticError(
                    "Edge endpoints contradict registered dependency roles"
                )

    @staticmethod
    def _snapshot_hash(snapshot) -> str:
        version, profile, components, edges = snapshot
        return canonical_hash(
            {
                "version": version,
                "profile": profile,
                "components": components,
                "edges": edges,
            }
        )

    def reconstruct(self, cutoff_utc: datetime):
        snapshot = self.registry.snapshot(
            deployment_pseudonym=self.contract.deployment_pseudonym,
            cutoff_utc=cutoff_utc,
        )
        _, _, components, edges = snapshot
        self._validate(components=components, edges=edges)
        return snapshot

    def _negative_controls(
        self,
        *,
        cutoff_utc: datetime,
    ) -> tuple[dict[str, Any], ...]:
        version, profile, components, edges = self.reconstruct(cutoff_utc)
        results = []
        mutable_edge = next(
            (
                item
                for item in edges
                if item.direction_semantics
                is not DirectionSemantics.BIDIRECTIONAL
                and not self._rules[
                    (item.dependency_type, item.direction_semantics)
                ].reversal_allowed
            ),
            None,
        )
        if mutable_edge is None:
            results.append(
                {
                    "mutation": "reversed_edge",
                    "rejected": False,
                    "reason": "no_nonreversible_edge",
                }
            )
        else:
            reversed_edge = mutable_edge.model_copy(
                update={
                    "source_component": (
                        mutable_edge.destination_component
                    ),
                    "destination_component": (
                        mutable_edge.source_component
                    ),
                }
            )
            mutated_edges = tuple(
                reversed_edge if item is mutable_edge else item
                for item in edges
            )
            try:
                self._validate(
                    components=components,
                    edges=mutated_edges,
                )
                rejected = False
                reason = "accepted"
            except TopologySemanticError as error:
                rejected = True
                reason = str(error)
            results.append(
                {
                    "mutation": "reversed_edge",
                    "rejected": rejected,
                    "reason": reason,
                }
            )

        stale_components = tuple(
            item.model_copy(update={"valid_to_utc": cutoff_utc})
            for item in components
        )
        stale_edges = tuple(
            item.model_copy(update={"valid_to_utc": cutoff_utc})
            for item in edges
        )
        stale_registry = TopologyRegistry(
            components=stale_components,
            edges=stale_edges,
            profiles=(profile,),
        )
        try:
            stale_registry.snapshot(
                deployment_pseudonym=self.contract.deployment_pseudonym,
                cutoff_utc=cutoff_utc,
            )
            stale_rejected = False
            stale_reason = "accepted"
        except ValueError as error:
            stale_rejected = True
            stale_reason = str(error)
        results.append(
            {
                "mutation": "stale_topology",
                "rejected": stale_rejected,
                "reason": stale_reason,
            }
        )
        if mutable_edge is not None:
            wrong_semantics = mutable_edge.model_copy(
                update={
                    "direction_semantics": (
                        DirectionSemantics.BIDIRECTIONAL
                    )
                }
            )
            wrong_edges = tuple(
                wrong_semantics if item is mutable_edge else item
                for item in edges
            )
            try:
                self._validate(components=components, edges=wrong_edges)
                semantics_rejected = False
                semantics_reason = "accepted"
            except TopologySemanticError as error:
                semantics_rejected = True
                semantics_reason = str(error)
            results.append(
                {
                    "mutation": "changed_direction_semantics",
                    "rejected": semantics_rejected,
                    "reason": semantics_reason,
                }
            )
        return tuple(results)

    def evaluate(
        self,
        *,
        cutoffs_utc: tuple[datetime, ...],
    ) -> TopologyAcceptanceReport:
        if not cutoffs_utc:
            raise ValueError("Topology acceptance requires decision cutoffs")
        results = []
        hashes = []
        versions = []
        deterministic_count = 0
        for cutoff in cutoffs_utc:
            first = self.reconstruct(cutoff)
            second = self.reconstruct(cutoff)
            first_hash = self._snapshot_hash(first)
            second_hash = self._snapshot_hash(second)
            deterministic = first_hash == second_hash
            deterministic_count += int(deterministic)
            hashes.append(first_hash)
            versions.append(first[0])
            results.append(
                {
                    "cutoff_utc": cutoff.isoformat(),
                    "snapshot_hash": first_hash,
                    "deterministic": deterministic,
                    "topology_version": first[0],
                }
            )
        mutations = self._negative_controls(cutoff_utc=cutoffs_utc[0])
        results.extend(mutations)
        change_epochs = sum(
            left != right
            for left, right in zip(versions, versions[1:], strict=False)
        )
        return TopologyAcceptanceReport(
            contract_hash=self.contract.content_hash(),
            cutoff_count=len(cutoffs_utc),
            reconstruction_fraction=(
                deterministic_count / len(cutoffs_utc)
            ),
            mutation_rejection_fraction=(
                sum(bool(item["rejected"]) for item in mutations)
                / len(mutations)
            ),
            snapshot_hashes=tuple(hashes),
            change_epoch_count=change_epochs,
            results=tuple(results),
        )
