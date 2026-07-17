from __future__ import annotations

import random

from .models import (
    DeploymentTopology,
    SimulationEdge,
    SimulationEdgeType,
    SimulationNode,
    SimulationNodeType,
)


def generate_hotel_topology(
    *,
    room_count: int,
    floor_count: int,
    redundant_middleware: bool,
    shared_storage: bool,
    seed: int,
) -> DeploymentTopology:
    if room_count < floor_count:
        raise ValueError(
            "room_count must be at least floor_count."
        )

    rng = random.Random(seed)
    storage_pool_count = 1 if shared_storage else 2
    total_storage_capacity_tb = (
        8.0 if room_count < 150 else 16.0
    )

    nodes: list[SimulationNode] = []
    edges: list[SimulationEdge] = []

    def add_node(
        node_id: str,
        node_type: SimulationNodeType,
        name: str,
        **kwargs: object,
    ) -> None:
        nodes.append(
            SimulationNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                **kwargs,
            )
        )

    def add_edge(
        source: str,
        target: str,
        edge_type: SimulationEdgeType,
        *,
        delay: int = 0,
        strength: float = 1.0,
    ) -> None:
        edges.append(
            SimulationEdge(
                source_node_id=source,
                target_node_id=target,
                edge_type=edge_type,
                propagation_delay_minutes=delay,
                propagation_strength=strength,
            )
        )

    add_node(
        "signal_source_1",
        SimulationNodeType.SIGNAL_SOURCE,
        "Primary signal source",
    )
    add_node(
        "gateway_1",
        SimulationNodeType.GATEWAY,
        "Primary IPTV gateway",
    )
    add_node(
        "streamer_1",
        SimulationNodeType.STREAMER,
        "Primary streamer",
    )
    add_node(
        "middleware_1",
        SimulationNodeType.MIDDLEWARE,
        "Primary middleware",
        capacity={
            "maximum_sessions": float(room_count * 2)
        },
    )
    add_node(
        "database_1",
        SimulationNodeType.DATABASE,
        "Middleware database",
    )
    for pool_index in range(1, storage_pool_count + 1):
        service_id = f"catchup_service_{pool_index}"
        storage_id = f"catchup_storage_{pool_index}"

        add_node(
            service_id,
            SimulationNodeType.CATCHUP_SERVICE,
            f"CatchUP service {pool_index}",
            metadata={
                "storage_scope": (
                    "shared" if shared_storage else "dedicated"
                ),
            },
        )
        add_node(
            storage_id,
            SimulationNodeType.CATCHUP_STORAGE,
            f"CatchUP storage {pool_index}",
            capacity={
                "usable_capacity_tb": (
                    total_storage_capacity_tb
                    / storage_pool_count
                )
            },
            configuration={
                "cleanup_enabled": True,
            },
            metadata={
                "shared": shared_storage,
                "dedicated_service_node_id": (
                    None if shared_storage else service_id
                ),
            },
        )
    add_node(
        "epg_service_1",
        SimulationNodeType.EPG_SERVICE,
        "EPG service",
    )
    add_node(
        "core_switch_1",
        SimulationNodeType.CORE_SWITCH,
        "Core IPTV switch",
    )

    add_edge(
        "signal_source_1",
        "gateway_1",
        SimulationEdgeType.SENDS_TO,
    )
    add_edge(
        "gateway_1",
        "streamer_1",
        SimulationEdgeType.SENDS_TO,
    )
    add_edge(
        "streamer_1",
        "middleware_1",
        SimulationEdgeType.SENDS_TO,
    )
    add_edge(
        "middleware_1",
        "database_1",
        SimulationEdgeType.DEPENDS_ON,
    )
    for pool_index in range(1, storage_pool_count + 1):
        service_id = f"catchup_service_{pool_index}"
        storage_id = f"catchup_storage_{pool_index}"

        add_edge(
            "middleware_1",
            service_id,
            SimulationEdgeType.DEPENDS_ON,
        )
        add_edge(
            service_id,
            storage_id,
            SimulationEdgeType.STORES_ON,
            delay=5,
        )
    add_edge(
        "middleware_1",
        "epg_service_1",
        SimulationEdgeType.DEPENDS_ON,
    )
    add_edge(
        "streamer_1",
        "core_switch_1",
        SimulationEdgeType.CONNECTED_TO,
    )

    if redundant_middleware:
        add_node(
            "middleware_2",
            SimulationNodeType.MIDDLEWARE,
            "Secondary middleware",
            capacity={
                "maximum_sessions": float(room_count * 2)
            },
            metadata={
                "redundant_peer": "middleware_1",
            },
        )
        add_edge(
            "middleware_2",
            "database_1",
            SimulationEdgeType.DEPENDS_ON,
        )
        for pool_index in range(1, storage_pool_count + 1):
            add_edge(
                "middleware_2",
                f"catchup_service_{pool_index}",
                SimulationEdgeType.DEPENDS_ON,
            )

    rooms_remaining = room_count

    for floor_index in range(1, floor_count + 1):
        floors_left = floor_count - floor_index + 1

        floor_rooms = (
            rooms_remaining // floors_left
            if floors_left > 1
            else rooms_remaining
        )

        rooms_remaining -= floor_rooms

        switch_id = f"distribution_switch_{floor_index}"
        client_id = f"smart_tv_group_{floor_index}"

        add_node(
            switch_id,
            SimulationNodeType.DISTRIBUTION_SWITCH,
            f"Distribution switch floor {floor_index}",
            capacity={
                "maximum_throughput_mbps": (
                    1000.0
                    if floor_rooms < 60
                    else 2000.0
                )
            },
            metadata={
                "floor": floor_index,
            },
        )
        add_node(
            client_id,
            SimulationNodeType.SMART_TV_GROUP,
            f"SmartTV group floor {floor_index}",
            metadata={
                "floor": floor_index,
                "room_count": floor_rooms,
                "device_cohort": rng.choice(
                    [
                        "lg_webos_a",
                        "lg_webos_b",
                        "samsung_tizen_a",
                    ]
                ),
            },
        )

        add_edge(
            "core_switch_1",
            switch_id,
            SimulationEdgeType.CONNECTED_TO,
            delay=1,
        )
        add_edge(
            switch_id,
            client_id,
            SimulationEdgeType.SERVES,
            delay=1,
        )

    return DeploymentTopology(
        deployment_family=(
            "small"
            if room_count < 100
            else "medium"
        ),
        room_count=room_count,
        floor_count=floor_count,
        shared_storage=shared_storage,
        nodes=nodes,
        edges=edges,
    )
