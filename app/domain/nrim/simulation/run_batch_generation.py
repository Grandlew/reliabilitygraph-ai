from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .batch_audit import audit_manifest
from .batch_generator import generate_dataset
from .dataset_manifest import (
    manifest_summary,
    save_manifest,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    output_dir = (
        base_dir.parent
        / "examples"
        / "simulation"
        / "day08_dataset_v2"
    )

    manifest = generate_dataset(
        output_dir=output_dir,
        start_time=datetime(
            2026,
            7,
            18,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        environment_count=72,
        ood_environment_count=12,
        generation_seed=42,
    )

    manifest_path = (
        output_dir / "manifest.json"
    )

    save_manifest(
        manifest=manifest,
        path=manifest_path,
    )

    summary = manifest_summary(manifest)
    audit = audit_manifest(manifest)

    audit_path = output_dir / "audit.json"

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
        )

    print("\nDATASET SUMMARY")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\nDATASET AUDIT")
    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\nManifest:", manifest_path)
    print("Audit:", audit_path)

    if audit["errors"]:
        raise RuntimeError(
            "Dataset audit found integrity errors."
        )


if __name__ == "__main__":
    main()
