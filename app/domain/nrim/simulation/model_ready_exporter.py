from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from .feature_schema import (
    FeatureSchema,
    SIGNAL_NAMES,
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


def add_causal_history_features(
    *,
    node_ids: list[str],
    feature_names: list[str],
    node_features: list[list[float]],
    history: dict[tuple[str, str], list[float]],
    persistence: dict[tuple[str, str], int],
) -> None:
    """Mutate rows using prior windows only, then update history."""
    index = {
        name: position
        for position, name in enumerate(feature_names)
    }
    pending: list[tuple[tuple[str, str], float, bool]] = []
    for node_id, row in zip(
        node_ids,
        node_features,
        strict=True,
    ):
        for signal_name in SIGNAL_NAMES:
            safe = signal_name.replace(".", "__")
            key = (node_id, safe)
            if f"{safe}__applicable" not in index:
                continue
            applicable = (
                row[index[f"{safe}__applicable"]] > 0.5
            )
            missing = row[index[f"{safe}__missing"]] > 0.5
            if not applicable or missing:
                continue
            current = float(row[index[f"{safe}__latest"]])
            previous = history[key]
            if previous:
                center = median(previous)
                deviations = [
                    abs(value - center)
                    for value in previous
                ]
                scale = max(
                    1e-6,
                    1.4826 * median(deviations),
                    abs(center) * 0.05,
                    1.0,
                )
                robust_z = (current - center) / scale
                delta = current - previous[-1]
            else:
                robust_z = 0.0
                delta = 0.0
            elevated = (
                robust_z >= 1.5
                or (
                    current > 0.0
                    and previous
                    and max(previous) <= 0.0
                )
            )
            current_persistence = (
                persistence[key] + 1
                if elevated
                else 0
            )
            row[index[f"history__{safe}__robust_z"]] = (
                robust_z
            )
            row[index[f"history__{safe}__delta"]] = delta
            row[
                index[
                    f"history__{safe}__persistence_count"
                ]
            ] = float(current_persistence)
            pending.append((key, current, elevated))

    for key, current, elevated in pending:
        history[key].append(current)
        # Bound memory while retaining a causal operational baseline.
        if len(history[key]) > 12:
            del history[key][:-12]
        persistence[key] = (
            persistence[key] + 1 if elevated else 0
        )


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
    history: dict[
        tuple[str, str],
        list[float],
    ] = defaultdict(list)
    persistence: dict[
        tuple[str, str],
        int,
    ] = defaultdict(int)

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
        node_feature_names = [
            definition.name
            for definition in feature_schema.node_features
        ]
        add_causal_history_features(
            node_ids=node_ids,
            feature_names=node_feature_names,
            node_features=node_features,
            history=history,
            persistence=persistence,
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
            node_feature_names=node_feature_names,
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
                separators=(",", ":"),
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
        "dataset_version": "0.4.1",
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
