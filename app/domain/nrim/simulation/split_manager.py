from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from .scenario_design import DatasetSplit


def canonical_topology_payload(
    topology: dict[str, Any],
) -> dict[str, Any]:
    nodes = sorted(
        [
            {
                "node_id": node["node_id"],
                "node_type": node["node_type"],
                "capacity": node.get(
                    "capacity",
                    {},
                ),
                "configuration": node.get(
                    "configuration",
                    {},
                ),
                "metadata": node.get(
                    "metadata",
                    {},
                ),
            }
            for node in topology.get("nodes", [])
        ],
        key=lambda item: item["node_id"],
    )

    edges = sorted(
        [
            {
                "source_node_id": edge[
                    "source_node_id"
                ],
                "target_node_id": edge[
                    "target_node_id"
                ],
                "edge_type": edge["edge_type"],
                "propagation_delay_minutes": edge.get(
                    "propagation_delay_minutes",
                    0,
                ),
                "propagation_strength": edge.get(
                    "propagation_strength",
                    1.0,
                ),
            }
            for edge in topology.get("edges", [])
        ],
        key=lambda item: (
            item["source_node_id"],
            item["target_node_id"],
            item["edge_type"],
        ),
    )

    return {
        "deployment_family": topology.get(
            "deployment_family"
        ),
        "room_count": topology.get("room_count"),
        "floor_count": topology.get("floor_count"),
        "nodes": nodes,
        "edges": edges,
    }


def topology_fingerprint(
    topology: dict[str, Any],
) -> str:
    payload = canonical_topology_payload(
        topology
    )

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:16]

    return f"topology_{digest}"


def deterministic_group_split(
    *,
    group_id: str,
    ood: bool = False,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetSplit:
    if ood:
        return DatasetSplit.OOD_TEST

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "Train plus validation fraction must be below 1."
        )

    digest = hashlib.sha256(
        group_id.encode("utf-8")
    ).hexdigest()

    value = int(digest[:12], 16) / float(
        16**12
    )

    if value < train_fraction:
        return DatasetSplit.TRAIN

    if value < (
        train_fraction + validation_fraction
    ):
        return DatasetSplit.VALIDATION

    return DatasetSplit.TEST


def validate_group_isolation(
    manifest_records: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    group_splits: dict[str, set[str]] = (
        defaultdict(set)
    )

    topology_splits: dict[str, set[str]] = (
        defaultdict(set)
    )

    for record in manifest_records:
        group_splits[
            str(record["pair_id"])
        ].add(str(record["split"]))

        topology_splits[
            str(record["topology_fingerprint"])
        ].add(str(record["split"]))

    for pair_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(
                f"Pair {pair_id} appears in "
                f"multiple splits: {sorted(splits)}"
            )

    for fingerprint, splits in (
        topology_splits.items()
    ):
        if len(splits) > 1:
            errors.append(
                f"Topology {fingerprint} appears in "
                f"multiple splits: {sorted(splits)}"
            )

    return errors
