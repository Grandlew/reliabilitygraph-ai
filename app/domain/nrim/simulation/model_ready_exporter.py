from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .feature_schema import (
    FeatureSchema,
    build_feature_schema,
)
from .graph_feature_builder import (
    build_edge_feature_matrix,
    build_node_feature_matrix,
)
from .sanitizer import (
    sanitize_observable_scenario,
)
from .target_builder import build_window_targets
from .temporal_windowing import (
    TemporalWindowSpecification,
    assert_no_future_events,
    events_in_observation_window,
    generate_window_boundaries,
    scenario_time_range,
)


class ModelReadyWindowRecord(BaseModel):
    window_id: str
    source_scenario_id: str

    split: str
    observation_start: str
    observation_cutoff: str
    prediction_end: str

    node_ids: list[str]
    node_feature_names: list[str]
    node_features: list[list[float]]

    edge_index: list[list[int]]
    edge_feature_names: list[str]
    edge_features: list[list[float]]

    targets: dict[str, Any]


class ModelReadyManifestRecord(BaseModel):
    window_id: str
    split: str
    path: str

    observation_hours: int
    prediction_horizon_hours: int

    node_count: int = Field(gt=0)
    edge_count: int = Field(ge=0)

    failure_type: str
    current_incident: int
    future_incident: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def export_scenario_windows(
    *,
    observable_path: Path,
    hidden_path: Path,
    split: str,
    output_dir: Path,
    specification: TemporalWindowSpecification,
    feature_schema: FeatureSchema | None = None,
) -> list[ModelReadyManifestRecord]:
    feature_schema = (
        feature_schema
        or build_feature_schema()
    )

    raw_observable = load_json(
        observable_path
    )
    hidden_scenario = load_json(hidden_path)

    sanitized = sanitize_observable_scenario(
        raw_observable
    )

    scenario_id = str(
        sanitized["scenario_id"]
    )

    scenario_start, scenario_end = (
        scenario_time_range(sanitized)
    )

    boundaries = generate_window_boundaries(
        scenario_id=scenario_id,
        scenario_start=scenario_start,
        scenario_end=scenario_end,
        specification=specification,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_records: list[
        ModelReadyManifestRecord
    ] = []

    for boundary in boundaries:
        telemetry = events_in_observation_window(
            events=sanitized.get(
                "telemetry",
                [],
            ),
            boundary=boundary,
        )

        context_events = (
            events_in_observation_window(
                events=sanitized.get(
                    "context_events",
                    [],
                ),
                boundary=boundary,
            )
        )

        if (
            len(telemetry)
            < specification.minimum_event_count
        ):
            continue

        assert_no_future_events(
            events=telemetry,
            cutoff=boundary.observation_cutoff,
        )

        assert_no_future_events(
            events=context_events,
            cutoff=boundary.observation_cutoff,
        )

        topology = sanitized["topology"]

        node_ids, node_features = (
            build_node_feature_matrix(
                topology=topology,
                telemetry=telemetry,
                context_events=context_events,
                cutoff=boundary.observation_cutoff,
                schema=feature_schema,
            )
        )

        edge_index, edge_features = (
            build_edge_feature_matrix(
                topology=topology,
                node_ids=node_ids,
                schema=feature_schema,
            )
        )

        targets = build_window_targets(
            node_ids=node_ids,
            hidden_scenario=hidden_scenario,
            boundary=boundary,
        )

        record = ModelReadyWindowRecord(
            window_id=boundary.window_id,
            source_scenario_id=scenario_id,
            split=split,
            observation_start=(
                boundary.observation_start.isoformat()
            ),
            observation_cutoff=(
                boundary.observation_cutoff.isoformat()
            ),
            prediction_end=(
                boundary.prediction_end.isoformat()
            ),
            node_ids=node_ids,
            node_feature_names=[
                definition.name
                for definition
                in feature_schema.node_features
            ],
            node_features=node_features,
            edge_index=edge_index,
            edge_feature_names=[
                definition.name
                for definition
                in feature_schema.edge_features
            ],
            edge_features=edge_features,
            targets=targets,
        )

        window_path = (
            output_dir
            / f"{boundary.window_id}.json"
        )

        with window_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                record.model_dump(mode="json"),
                file,
                indent=2,
            )

        manifest_records.append(
            ModelReadyManifestRecord(
                window_id=boundary.window_id,
                split=split,
                path=str(window_path),
                observation_hours=(
                    specification.observation_hours
                ),
                prediction_horizon_hours=(
                    specification.prediction_horizon_hours
                ),
                node_count=len(node_ids),
                edge_count=len(edge_index),
                failure_type=str(
                    targets["failure_type"]
                ),
                current_incident=int(
                    targets["current_incident"]
                ),
                future_incident=int(
                    targets["future_incident"]
                ),
            )
        )

    return manifest_records


def export_dataset(
    *,
    day08_manifest_path: Path,
    output_root: Path,
    specification: TemporalWindowSpecification,
) -> dict[str, Any]:
    manifest = load_json(
        day08_manifest_path
    )

    feature_schema = build_feature_schema()

    all_records: list[
        ModelReadyManifestRecord
    ] = []

    for raw_record in manifest["records"]:
        split = str(raw_record["split"])

        observable_path = Path(
            raw_record["observable_path"]
        )
        hidden_path = Path(
            raw_record["hidden_path"]
        )

        split_output_dir = output_root / split

        records = export_scenario_windows(
            observable_path=observable_path,
            hidden_path=hidden_path,
            split=split,
            output_dir=split_output_dir,
            specification=specification,
            feature_schema=feature_schema,
        )

        all_records.extend(records)

    exported_manifest = {
        "dataset_name": (
            "nrim_model_ready_temporal_graphs"
        ),
        "dataset_version": "0.1.0",
        "source_dataset_version": (
            manifest.get("dataset_version")
        ),
        "window_specification": (
            specification.model_dump(mode="json")
        ),
        "feature_schema": (
            feature_schema.model_dump(mode="json")
        ),
        "records": [
            record.model_dump(mode="json")
            for record in all_records
        ],
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_root / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            exported_manifest,
            file,
            indent=2,
        )

    return exported_manifest
