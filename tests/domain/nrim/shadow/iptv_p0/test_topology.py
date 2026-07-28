from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.topology import (
    CaptureMode,
    IptvEdge,
    IptvEdgeType,
    IptvNodeType,
    TopologyCapture,
    TopologyHistory,
    validate_topology,
)

BASE_TIME = datetime(
    2026,
    1,
    2,
    12,
    0,
    tzinfo=timezone.utc,
)
DEPLOYMENT = "dep_" + "a" * 24


def test_full_topology_reconstructs_deterministically(topology_capture):
    history = TopologyHistory((topology_capture,))
    first = history.reconstruct(
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )
    second = history.reconstruct(
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )
    assert first.content_hash == second.content_hash
    assert first.applied_capture_ids == (topology_capture.capture_id,)


def test_exact_bitemporal_cutoff_includes_capture(topology_capture):
    result = TopologyHistory((topology_capture,)).reconstruct(
        event_cutoff_utc=topology_capture.effective_at_utc,
        knowledge_cutoff_utc=topology_capture.recorded_at_utc,
    )
    assert result.topology_version == topology_capture.topology_version


@pytest.mark.parametrize("cutoff_kind", ("event", "knowledge"))
def test_topology_rejects_future_information(topology_capture, cutoff_kind):
    event = BASE_TIME
    knowledge = BASE_TIME
    if cutoff_kind == "event":
        event = topology_capture.effective_at_utc - timedelta(microseconds=1)
    else:
        knowledge = topology_capture.recorded_at_utc - timedelta(
            microseconds=1
        )
    with pytest.raises(ValueError, match="No topology"):
        TopologyHistory((topology_capture,)).reconstruct(
            event_cutoff_utc=event,
            knowledge_cutoff_utc=knowledge,
        )


def test_delta_addition_emits_new_epoch(topology_capture):
    added = topology_capture.nodes[0].model_copy(
        update={"node_id": "topo_" + "f" * 12}
    )
    delta = TopologyCapture(
        capture_id="capture_" + "b" * 12,
        deployment_pseudonym=DEPLOYMENT,
        topology_version="topology-2",
        mode=CaptureMode.DELTA,
        effective_at_utc=BASE_TIME + timedelta(minutes=1),
        recorded_at_utc=BASE_TIME + timedelta(minutes=2),
        nodes=(added,),
    )
    result = TopologyHistory((topology_capture, delta)).reconstruct(
        event_cutoff_utc=BASE_TIME + timedelta(minutes=2),
        knowledge_cutoff_utc=BASE_TIME + timedelta(minutes=2),
    )
    assert result.topology_version == "topology-2"
    assert added in result.nodes
    assert len(result.applied_capture_ids) == 2


def test_delta_removal_removes_incident_edges(topology_capture):
    removed = topology_capture.nodes[-1]
    delta = TopologyCapture(
        capture_id="capture_" + "b" * 12,
        deployment_pseudonym=DEPLOYMENT,
        topology_version="topology-2",
        mode=CaptureMode.DELTA,
        effective_at_utc=BASE_TIME + timedelta(minutes=1),
        recorded_at_utc=BASE_TIME + timedelta(minutes=2),
        removed_node_ids=(removed.node_id,),
    )
    result = TopologyHistory((topology_capture, delta)).reconstruct(
        event_cutoff_utc=BASE_TIME + timedelta(minutes=2),
        knowledge_cutoff_utc=BASE_TIME + timedelta(minutes=2),
    )
    assert removed not in result.nodes
    assert all(
        removed.node_id
        not in {edge.source_node_id, edge.destination_node_id}
        for edge in result.edges
    )


@pytest.mark.parametrize("edge_index", (0, 1, 2, 3, 4))
def test_forward_service_path_edges_reject_reversal(
    topology_capture,
    edge_index,
):
    edge = topology_capture.edges[edge_index]
    reversed_edge = edge.model_copy(
        update={
            "source_node_id": edge.destination_node_id,
            "destination_node_id": edge.source_node_id,
        }
    )
    edges = tuple(
        reversed_edge if item.edge_id == edge.edge_id else item
        for item in topology_capture.edges
    )
    with pytest.raises(ValueError, match="Reversed|invalid"):
        validate_topology(topology_capture.nodes, edges)


def test_membership_direction_is_access_to_group(topology_capture):
    edge = next(
        item
        for item in topology_capture.edges
        if item.edge_type is IptvEdgeType.MEMBERSHIP
    )
    invalid = edge.model_copy(
        update={
            "source_node_id": topology_capture.nodes[6].node_id,
            "destination_node_id": topology_capture.nodes[4].node_id,
        }
    )
    edges = tuple(
        invalid if item.edge_id == edge.edge_id else item
        for item in topology_capture.edges
    )
    with pytest.raises(ValueError, match="Membership"):
        validate_topology(topology_capture.nodes, edges)


def test_control_edge_requires_multicast_control_source(topology_capture):
    edge = next(
        item
        for item in topology_capture.edges
        if item.edge_type is IptvEdgeType.CONTROLS
    )
    invalid = edge.model_copy(
        update={"source_node_id": topology_capture.nodes[1].node_id}
    )
    edges = tuple(
        invalid if item.edge_id == edge.edge_id else item
        for item in topology_capture.edges
    )
    with pytest.raises(ValueError, match="Control"):
        validate_topology(topology_capture.nodes, edges)


def test_multicast_edge_requires_group_identity(topology_capture):
    edge = next(
        item
        for item in topology_capture.edges
        if item.edge_type is IptvEdgeType.MULTICASTS_TO
    )
    value = edge.model_dump(mode="json")
    value["multicast_group_id"] = None
    with pytest.raises(ValidationError):
        IptvEdge.model_validate(value)


def test_full_capture_cannot_encode_removals(topology_capture):
    value = topology_capture.model_dump(mode="json")
    value["removed_node_ids"] = (topology_capture.nodes[0].node_id,)
    with pytest.raises(ValidationError):
        TopologyCapture.model_validate(value)


def test_history_rejects_delta_without_full_baseline(topology_capture):
    value = topology_capture.model_dump(mode="json")
    value["mode"] = CaptureMode.DELTA
    delta = TopologyCapture.model_validate(value)
    with pytest.raises(ValueError, match="full baseline"):
        TopologyHistory((delta,)).reconstruct(
            event_cutoff_utc=BASE_TIME,
            knowledge_cutoff_utc=BASE_TIME,
        )


def test_history_rejects_multiple_deployments(topology_capture):
    other = topology_capture.model_copy(
        update={
            "capture_id": "capture_" + "c" * 12,
            "deployment_pseudonym": "dep_" + "c" * 24,
        }
    )
    with pytest.raises(ValueError, match="one deployment"):
        TopologyHistory((topology_capture, other))


def test_history_rejects_duplicate_capture_identity(topology_capture):
    with pytest.raises(ValueError, match="identities"):
        TopologyHistory((topology_capture, topology_capture))


def test_topology_node_types_cover_iptv_p0_path(topology_capture):
    assert {item.node_type for item in topology_capture.nodes} == set(
        IptvNodeType
    )
