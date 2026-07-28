from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..hashing import canonical_hash
from ..privacy import assert_no_direct_identifiers
from .contracts import utc


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class IptvNodeType(str, Enum):
    SOURCE_HEADEND = "source_headend"
    CORE = "core"
    AGGREGATION = "aggregation"
    MULTICAST_CONTROL = "multicast_control"
    MULTICAST_GROUP = "multicast_group"
    TRANSPORT = "transport"
    ACCESS_HANDOFF = "access_handoff"
    CUSTOMER_IMPACT = "customer_impact"


class IptvEdgeType(str, Enum):
    FEEDS = "feeds"
    TRANSPORTS_TO = "transports_to"
    MULTICASTS_TO = "multicasts_to"
    MEMBERSHIP = "membership"
    CONTROLS = "controls"
    AFFECTS = "affects"


class CaptureMode(str, Enum):
    FULL = "full"
    DELTA = "delta"


class IptvNode(StrictModel):
    node_id: str = Field(pattern=r"^topo_[0-9a-f]{8,64}$")
    node_type: IptvNodeType
    service_path_id: str = Field(pattern=r"^path_[0-9a-f]{8,64}$")


class IptvEdge(StrictModel):
    edge_id: str = Field(pattern=r"^edge_[0-9a-f]{8,64}$")
    source_node_id: str = Field(pattern=r"^topo_[0-9a-f]{8,64}$")
    destination_node_id: str = Field(pattern=r"^topo_[0-9a-f]{8,64}$")
    edge_type: IptvEdgeType
    direction: str = "source_to_destination"
    multicast_group_id: str | None = Field(
        default=None,
        pattern=r"^group_[0-9a-f]{8,64}$",
    )

    @model_validator(mode="after")
    def valid_edge(self) -> Self:
        if self.source_node_id == self.destination_node_id:
            raise ValueError("IPTV topology self-loops are forbidden")
        if self.direction != "source_to_destination":
            raise ValueError("Topology direction must be explicit")
        if (
            self.edge_type
            in {IptvEdgeType.MULTICASTS_TO, IptvEdgeType.MEMBERSHIP}
            and self.multicast_group_id is None
        ):
            raise ValueError("Multicast relationships require a group")
        return self


class TopologyCapture(StrictModel):
    capture_id: str = Field(pattern=r"^capture_[0-9a-f]{8,64}$")
    deployment_pseudonym: str = Field(
        pattern=r"^dep_[0-9a-f]{16,64}$"
    )
    topology_version: str = Field(min_length=1, max_length=128)
    mode: CaptureMode
    effective_at_utc: datetime
    recorded_at_utc: datetime
    nodes: tuple[IptvNode, ...] = ()
    edges: tuple[IptvEdge, ...] = ()
    removed_node_ids: tuple[str, ...] = ()
    removed_edge_ids: tuple[str, ...] = ()

    @field_validator("effective_at_utc", "recorded_at_utc")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        node_ids = [item.node_id for item in self.nodes]
        edge_ids = [item.edge_id for item in self.edges]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Capture contains duplicate nodes")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Capture contains duplicate edges")
        if self.mode is CaptureMode.FULL and (
            self.removed_node_ids or self.removed_edge_ids
        ):
            raise ValueError("Full captures cannot contain removals")
        assert_no_direct_identifiers(self.model_dump(mode="json"))
        return self


_PATH_RANK = {
    IptvNodeType.SOURCE_HEADEND: 0,
    IptvNodeType.CORE: 1,
    IptvNodeType.AGGREGATION: 2,
    IptvNodeType.TRANSPORT: 3,
    IptvNodeType.ACCESS_HANDOFF: 4,
    IptvNodeType.CUSTOMER_IMPACT: 5,
}


def validate_topology(
    nodes: tuple[IptvNode, ...],
    edges: tuple[IptvEdge, ...],
) -> None:
    node_by_id = {item.node_id: item for item in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("Topology node identities are not unique")
    if len({item.edge_id for item in edges}) != len(edges):
        raise ValueError("Topology edge identities are not unique")
    for edge in edges:
        if (
            edge.source_node_id not in node_by_id
            or edge.destination_node_id not in node_by_id
        ):
            raise ValueError("Topology edge references an unknown node")
        source = node_by_id[edge.source_node_id].node_type
        destination = node_by_id[edge.destination_node_id].node_type
        if edge.edge_type in {
            IptvEdgeType.FEEDS,
            IptvEdgeType.TRANSPORTS_TO,
            IptvEdgeType.AFFECTS,
        }:
            if source not in _PATH_RANK or destination not in _PATH_RANK:
                raise ValueError("Service-path edge has invalid endpoint type")
            if _PATH_RANK[source] >= _PATH_RANK[destination]:
                raise ValueError("Reversed or non-forward service-path edge")
        elif edge.edge_type is IptvEdgeType.CONTROLS:
            if source is not IptvNodeType.MULTICAST_CONTROL:
                raise ValueError("Control edges must originate at control")
        elif edge.edge_type is IptvEdgeType.MULTICASTS_TO:
            if destination is not IptvNodeType.MULTICAST_GROUP:
                raise ValueError("Multicast edge must target a group")
        elif edge.edge_type is IptvEdgeType.MEMBERSHIP:
            if (
                source is not IptvNodeType.ACCESS_HANDOFF
                or destination is not IptvNodeType.MULTICAST_GROUP
            ):
                raise ValueError("Membership must link access to group")


class TopologySnapshot(StrictModel):
    deployment_pseudonym: str
    event_cutoff_utc: datetime
    knowledge_cutoff_utc: datetime
    applied_capture_ids: tuple[str, ...]
    topology_version: str
    nodes: tuple[IptvNode, ...]
    edges: tuple[IptvEdge, ...]

    @field_validator("event_cutoff_utc", "knowledge_cutoff_utc")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return utc(value)

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class TopologyHistory:
    def __init__(self, captures: tuple[TopologyCapture, ...]) -> None:
        if not captures:
            raise ValueError("Topology history requires captures")
        deployments = {item.deployment_pseudonym for item in captures}
        if len(deployments) != 1:
            raise ValueError("Topology history must bind one deployment")
        if len({item.capture_id for item in captures}) != len(captures):
            raise ValueError("Topology capture identities must be unique")
        self._captures = tuple(
            sorted(
                captures,
                key=lambda item: (
                    item.effective_at_utc,
                    item.recorded_at_utc,
                    item.capture_id,
                ),
            )
        )
        self.deployment_pseudonym = next(iter(deployments))

    def reconstruct(
        self,
        *,
        event_cutoff_utc: datetime,
        knowledge_cutoff_utc: datetime,
    ) -> TopologySnapshot:
        event_cutoff = utc(event_cutoff_utc)
        knowledge_cutoff = utc(knowledge_cutoff_utc)
        applicable = tuple(
            item
            for item in self._captures
            if item.effective_at_utc <= event_cutoff
            and item.recorded_at_utc <= knowledge_cutoff
        )
        if not applicable:
            raise ValueError("No topology was known at the requested cutoffs")
        if applicable[0].mode is not CaptureMode.FULL:
            raise ValueError("Topology history lacks a known full baseline")
        nodes: dict[str, IptvNode] = {}
        edges: dict[str, IptvEdge] = {}
        for capture in applicable:
            if capture.mode is CaptureMode.FULL:
                nodes = {item.node_id: item for item in capture.nodes}
                edges = {item.edge_id: item for item in capture.edges}
            else:
                for edge_id in capture.removed_edge_ids:
                    edges.pop(edge_id, None)
                for node_id in capture.removed_node_ids:
                    nodes.pop(node_id, None)
                    edges = {
                        key: edge
                        for key, edge in edges.items()
                        if node_id
                        not in {
                            edge.source_node_id,
                            edge.destination_node_id,
                        }
                    }
                nodes.update({item.node_id: item for item in capture.nodes})
                edges.update({item.edge_id: item for item in capture.edges})
            validate_topology(
                tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
                tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
            )
        latest = applicable[-1]
        return TopologySnapshot(
            deployment_pseudonym=self.deployment_pseudonym,
            event_cutoff_utc=event_cutoff,
            knowledge_cutoff_utc=knowledge_cutoff,
            applied_capture_ids=tuple(item.capture_id for item in applicable),
            topology_version=latest.topology_version,
            nodes=tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
            edges=tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
        )


def load_topology_jsonl(
    path: Path,
    *,
    allowed_root: Path,
) -> TopologyHistory:
    root = allowed_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Topology input escapes the approved offline root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() != ".jsonl":
        raise ValueError("Topology capture input must be JSONL")
    if resolved.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Topology capture input exceeds the size limit")
    captures = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Topology capture line {line_number} is empty"
                )
            try:
                value = json.loads(line)
                capture = TopologyCapture.model_validate(value)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Topology capture line {line_number} is invalid"
                ) from error
            captures.append(capture)
    return TopologyHistory(tuple(captures))
