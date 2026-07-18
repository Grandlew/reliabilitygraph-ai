from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sanitizer import (
    FORBIDDEN_RECURSIVE_KEYS,
    FORBIDDEN_TOP_LEVEL_KEYS,
)
from .temporal_windowing import parse_timestamp


ADDITIONAL_FORBIDDEN_KEYS = {
    "environment_id",
    "pair_id",
    "simulation_metadata",
    "confounder_type",
    "model_visible",
    "simulated_transient",
    "topology_seed",
    "workload_seed",
    "observation_seed",
    "fault_seed",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def recursively_find_forbidden_keys(
    value: Any,
    *,
    path: str = "root",
) -> list[str]:
    forbidden = (
        FORBIDDEN_RECURSIVE_KEYS
        | FORBIDDEN_TOP_LEVEL_KEYS
        | ADDITIONAL_FORBIDDEN_KEYS
    )

    errors: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"

            if key in forbidden:
                errors.append(
                    f"Forbidden key at {child_path}."
                )

            errors.extend(
                recursively_find_forbidden_keys(
                    child,
                    path=child_path,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(
                recursively_find_forbidden_keys(
                    child,
                    path=f"{path}[{index}]",
                )
            )

    return errors


def audit_window_record(
    record: dict[str, Any],
) -> list[str]:
    errors = recursively_find_forbidden_keys(
        record
    )

    window_id = str(
        record.get("window_id", "")
    )

    forbidden_tokens = {
        "healthy",
        "storage",
        "cleanup",
        "worker",
        "failure",
        "fault",
    }

    lowered_window_id = window_id.lower()

    for token in forbidden_tokens:
        if token in lowered_window_id:
            errors.append(
                "Window ID appears to encode a target class."
            )
            break

    node_ids = record.get("node_ids", [])
    node_features = record.get(
        "node_features",
        [],
    )

    if len(node_ids) != len(node_features):
        errors.append(
            "Node feature row count does not match node count."
        )

    edge_index = record.get(
        "edge_index",
        [],
    )
    edge_features = record.get(
        "edge_features",
        [],
    )

    if len(edge_index) != len(edge_features):
        errors.append(
            "Edge feature row count does not match edge count."
        )

    targets = record.get("targets", {})

    root_labels = targets.get(
        "root_cause_node",
        [],
    )

    if len(root_labels) != len(node_ids):
        errors.append(
            "Root-cause target length does not match node count."
        )

    affected_labels = targets.get(
        "affected_service_node",
        [],
    )

    if len(affected_labels) != len(node_ids):
        errors.append(
            "Affected-service target length does not match node count."
        )

    return errors


def audit_model_ready_dataset(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    split_counts = Counter()
    failure_counts = Counter()
    future_incident_counts = Counter()

    seen_window_ids: set[str] = set()

    for manifest_record in manifest.get(
        "records",
        [],
    ):
        window_id = str(
            manifest_record["window_id"]
        )

        if window_id in seen_window_ids:
            errors.append(
                f"Duplicate window ID: {window_id}"
            )

        seen_window_ids.add(window_id)

        split = str(
            manifest_record["split"]
        )

        split_counts[split] += 1
        failure_counts[
            str(
                manifest_record[
                    "failure_type"
                ]
            )
        ] += 1
        future_incident_counts[
            str(
                manifest_record[
                    "future_incident"
                ]
            )
        ] += 1

        path = Path(
            manifest_record["path"]
        )

        record = load_json(path)

        window_errors = audit_window_record(
            record
        )

        errors.extend(
            [
                f"{window_id}: {error}"
                for error in window_errors
            ]
        )

        cutoff = parse_timestamp(
            record["observation_cutoff"]
        )

        prediction_end = parse_timestamp(
            record["prediction_end"]
        )

        if cutoff >= prediction_end:
            errors.append(
                f"{window_id}: invalid prediction interval."
            )

    if not manifest.get("records"):
        errors.append(
            "Model-ready dataset contains no windows."
        )

    if len(future_incident_counts) < 2:
        warnings.append(
            "Future-incident target contains only one class."
        )

    return {
        "window_count": len(
            manifest.get("records", [])
        ),
        "split_counts": dict(split_counts),
        "failure_type_counts": dict(
            failure_counts
        ),
        "future_incident_counts": dict(
            future_incident_counts
        ),
        "errors": errors,
        "warnings": warnings,
    }
