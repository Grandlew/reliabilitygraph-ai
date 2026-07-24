from __future__ import annotations

import json
from pathlib import Path

from .model_ready_audit import (
    audit_model_ready_dataset,
)
from .model_ready_exporter import export_dataset
from .temporal_windowing import (
    TemporalWindowSpecification,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    simulation_root = (
        base_dir.parent
        / "examples"
        / "simulation"
    )
    source_root = (
        simulation_root / "day08_dataset_v06"
    )
    output_root = (
        simulation_root / "day09_model_ready_v06"
    )
    if output_root.exists() and any(
        output_root.iterdir()
    ):
        raise RuntimeError(
            "Refusing to overwrite an existing v0.6 model-ready dataset"
        )
    manifest = export_dataset(
        day08_manifest_path=(
            source_root / "manifest.json"
        ),
        output_root=output_root,
        specification=TemporalWindowSpecification(
            observation_hours=6,
            prediction_horizon_hours=6,
            stride_hours=2,
            minimum_event_count=5,
        ),
    )
    audit = audit_model_ready_dataset(
        manifest
    )
    audit_path = output_root / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "window_count": audit["window_count"],
                "split_counts": audit["split_counts"],
                "errors": audit["errors"],
                "warnings": audit["warnings"],
            },
            indent=2,
        )
    )
    if audit["errors"]:
        raise RuntimeError(
            "v0.6 model-ready audit found errors"
        )


if __name__ == "__main__":
    main()

