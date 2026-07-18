from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .scenario_design import DatasetSplit
from .split_manager import validate_group_isolation


class ScenarioManifestRecord(BaseModel):
    scenario_id: str
    pair_id: str
    environment_id: str

    topology_fingerprint: str
    split: DatasetSplit

    failure_type: str
    healthy: bool

    room_count: int = Field(gt=0)
    floor_count: int = Field(gt=0)

    fault_severity: float | None = None
    fault_injection_fraction: float | None = None

    confounders: list[str] = Field(
        default_factory=list
    )
    missingness_mode: str
    missing_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    observable_path: str
    hidden_path: str

    topology_seed: int
    workload_seed: int
    observation_seed: int
    fault_seed: int


class DatasetManifest(BaseModel):
    dataset_name: str
    dataset_version: str
    simulator_version: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    generation_seed: int
    records: list[ScenarioManifestRecord] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_manifest(
        self,
    ) -> "DatasetManifest":
        scenario_ids = [
            record.scenario_id
            for record in self.records
        ]

        if len(scenario_ids) != len(
            set(scenario_ids)
        ):
            raise ValueError(
                "Manifest contains duplicate scenario IDs."
            )

        isolation_errors = validate_group_isolation(
            [
                record.model_dump(mode="json")
                for record in self.records
            ]
        )

        if isolation_errors:
            raise ValueError(
                "Dataset split isolation failed: "
                + "; ".join(isolation_errors)
            )

        return self


def manifest_summary(
    manifest: DatasetManifest,
) -> dict[str, Any]:
    split_counts = Counter(
        record.split.value
        for record in manifest.records
    )

    failure_counts = Counter(
        record.failure_type
        for record in manifest.records
    )

    topology_counts = Counter(
        record.topology_fingerprint
        for record in manifest.records
    )

    return {
        "scenario_count": len(manifest.records),
        "split_counts": dict(split_counts),
        "failure_type_counts": dict(
            failure_counts
        ),
        "unique_topology_count": len(
            topology_counts
        ),
        "healthy_count": sum(
            1
            for record in manifest.records
            if record.healthy
        ),
        "faulty_count": sum(
            1
            for record in manifest.records
            if not record.healthy
        ),
    }


def save_manifest(
    *,
    manifest: DatasetManifest,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest.model_dump(mode="json"),
            file,
            indent=2,
        )
