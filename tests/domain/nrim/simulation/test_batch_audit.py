from datetime import datetime, timezone

from app.domain.nrim.simulation.batch_audit import (
    audit_manifest,
)
from app.domain.nrim.simulation.batch_generator import (
    generate_dataset,
)


def test_generated_dataset_has_no_integrity_errors(
    tmp_path,
) -> None:
    manifest = generate_dataset(
        output_dir=tmp_path,
        start_time=datetime(
            2026,
            7,
            18,
            tzinfo=timezone.utc,
        ),
        environment_count=3,
        ood_environment_count=1,
        generation_seed=42,
    )

    audit = audit_manifest(manifest)

    assert audit["errors"] == []


def test_audit_reports_expected_scenario_count(
    tmp_path,
) -> None:
    manifest = generate_dataset(
        output_dir=tmp_path,
        start_time=datetime(
            2026,
            7,
            18,
            tzinfo=timezone.utc,
        ),
        environment_count=2,
        ood_environment_count=1,
        generation_seed=42,
    )

    audit = audit_manifest(manifest)

    assert audit["scenario_count"] == 6
