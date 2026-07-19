from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

from .models import WindowSummary


MISSING_FEATURE_SUFFIX = "__missing"


def load_json(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_model_ready_manifest(
    path: str | Path,
) -> dict[str, Any]:
    manifest = load_json(path)

    records = manifest.get("records")

    if not isinstance(records, list):
        raise ValueError(
            "Model-ready manifest has no records list."
        )

    feature_schema = manifest.get("feature_schema")

    if not isinstance(feature_schema, dict):
        raise ValueError(
            "Model-ready manifest has no feature schema."
        )

    return manifest


def iter_window_records(
    manifest: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for manifest_record in manifest["records"]:
        window_path = Path(
            str(manifest_record["path"])
        )

        yield manifest_record, load_json(window_path)


def validate_numeric_matrix(
    matrix: list[list[Any]],
    *,
    expected_columns: int,
    matrix_name: str,
) -> None:
    for row_index, row in enumerate(matrix):
        if len(row) != expected_columns:
            raise ValueError(
                f"{matrix_name} row {row_index} has "
                f"{len(row)} columns; expected "
                f"{expected_columns}."
            )

        for column_index, value in enumerate(row):
            if isinstance(value, bool):
                continue

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{matrix_name}[{row_index}]"
                    f"[{column_index}] is not numeric."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{matrix_name}[{row_index}]"
                    f"[{column_index}] is not finite."
                )


def missing_feature_fraction(
    *,
    feature_names: list[str],
    matrix: list[list[float]],
) -> float:
    missing_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.endswith(MISSING_FEATURE_SUFFIX)
    ]

    if not missing_indices or not matrix:
        return 0.0

    values = [
        float(row[index])
        for row in matrix
        for index in missing_indices
    ]

    return sum(values) / len(values)


def summarize_window(
    window: dict[str, Any],
) -> WindowSummary:
    node_ids = list(window["node_ids"])
    node_feature_names = list(
        window["node_feature_names"]
    )
    node_features = list(window["node_features"])

    edge_index = list(window["edge_index"])
    edge_feature_names = list(
        window["edge_feature_names"]
    )
    edge_features = list(window["edge_features"])

    if len(node_ids) != len(node_features):
        raise ValueError(
            "Node count and node-feature rows differ."
        )

    if len(edge_index) != len(edge_features):
        raise ValueError(
            "Edge count and edge-feature rows differ."
        )

    validate_numeric_matrix(
        node_features,
        expected_columns=len(node_feature_names),
        matrix_name="node_features",
    )

    validate_numeric_matrix(
        edge_features,
        expected_columns=len(edge_feature_names),
        matrix_name="edge_features",
    )

    targets = window["targets"]

    root_targets = list(
        targets["root_cause_node"]
    )
    affected_targets = list(
        targets["affected_service_node"]
    )

    if len(root_targets) != len(node_ids):
        raise ValueError(
            "Root-cause target length differs from node count."
        )

    if len(affected_targets) != len(node_ids):
        raise ValueError(
            "Affected-service target length differs "
            "from node count."
        )

    return WindowSummary(
        window_id=str(window["window_id"]),
        split=str(window["split"]),
        path="",
        node_count=len(node_ids),
        edge_count=len(edge_index),
        node_feature_count=len(node_feature_names),
        edge_feature_count=len(edge_feature_names),
        failure_type=str(targets["failure_type"]),
        current_incident=int(
            targets["current_incident"]
        ),
        future_incident=int(
            targets["future_incident"]
        ),
        root_cause_positive_count=sum(
            int(value)
            for value in root_targets
        ),
        affected_service_positive_count=sum(
            int(value)
            for value in affected_targets
        ),
        missing_feature_fraction=round(
            missing_feature_fraction(
                feature_names=node_feature_names,
                matrix=node_features,
            ),
            6,
        ),
    )


def load_window_summaries(
    manifest: dict[str, Any],
) -> list[WindowSummary]:
    summaries: list[WindowSummary] = []

    for manifest_record, window in iter_window_records(
        manifest
    ):
        summary = summarize_window(window)

        summaries.append(
            summary.model_copy(
                update={
                    "path": str(
                        manifest_record["path"]
                    ),
                }
            )
        )

    return summaries
