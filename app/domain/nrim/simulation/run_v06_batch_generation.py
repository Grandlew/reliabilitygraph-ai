from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .batch_audit import audit_manifest
from .batch_generator import (
    V06_TOPOLOGY_GROUP_COUNTS,
    generate_dataset,
)
from .dataset_manifest import (
    manifest_summary,
    save_manifest,
)
from .locked_test_governance import (
    create_locked_test_seal,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = (
        base_dir.parent
        / "examples"
        / "simulation"
        / "day08_dataset_v06"
    )
    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        raise RuntimeError(
            "Refusing to overwrite an existing v0.6 dataset"
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
        environment_count=0,
        ood_environment_count=0,
        generation_seed=606,
        topology_group_counts=(
            V06_TOPOLOGY_GROUP_COUNTS
        ),
        dataset_version="0.4.1",
        simulator_version="0.5.1",
    )
    manifest_path = output_dir / "manifest.json"
    save_manifest(
        manifest=manifest,
        path=manifest_path,
    )
    audit = audit_manifest(manifest)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    if audit["errors"]:
        raise RuntimeError(
            "v0.6 dataset audit found integrity errors"
        )
    seal = create_locked_test_seal(
        manifest=manifest,
        output_dir=output_dir / "governance",
    )
    summary = manifest_summary(manifest)
    print(json.dumps(summary, indent=2))
    print(
        "Locked-test commitment:",
        seal["commitment_sha256"],
    )
    print("Manifest:", manifest_path)
    print("Audit:", audit_path)


if __name__ == "__main__":
    main()
