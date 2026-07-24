from __future__ import annotations

import json
from pathlib import Path

from .model_ready_audit import (
    audit_model_ready_dataset,
)
from .model_ready_exporter import (
    export_dataset,
)
from .temporal_windowing import (
    TemporalWindowSpecification,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    day08_root = (
        base_dir.parent
        / "examples"
        / "simulation"
        / "day08_dataset_v2"
    )

    day09_root = (
        base_dir.parent
        / "examples"
        / "simulation"
        / "day09_model_ready_v2"
    )

    specification = TemporalWindowSpecification(
        observation_hours=6,
        prediction_horizon_hours=6,
        stride_hours=2,
        minimum_event_count=5,
    )

    manifest = export_dataset(
        day08_manifest_path=(
            day08_root / "manifest.json"
        ),
        output_root=day09_root,
        specification=specification,
    )

    audit = audit_model_ready_dataset(
        manifest
    )

    audit_path = day09_root / "audit.json"

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
        )

    print("\nMODEL-READY DATASET")
    print(
        json.dumps(
            {
                "window_count": audit[
                    "window_count"
                ],
                "split_counts": audit[
                    "split_counts"
                ],
                "failure_type_counts": audit[
                    "failure_type_counts"
                ],
                "future_incident_counts": audit[
                    "future_incident_counts"
                ],
            },
            indent=2,
        )
    )

    print("\nAUDIT")
    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print(
        "\nManifest:",
        day09_root / "manifest.json",
    )
    print("Audit:", audit_path)

    if audit["errors"]:
        raise RuntimeError(
            "Model-ready export contains audit errors."
        )


if __name__ == "__main__":
    main()
