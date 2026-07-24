from datetime import datetime, timezone

from app.domain.nrim.simulation.batch_generator import (
    balanced_hard_negative_confounders,
    balanced_environment_assignments,
    create_counterfactual_plans,
    generate_dataset,
)
from app.domain.nrim.simulation.scenario_design import (
    ConfounderType,
    DatasetSplit,
)
from app.domain.nrim.simulation.models import (
    FailureType,
)


def test_counterfactual_plans_share_environment() -> None:
    faulty, healthy = create_counterfactual_plans(
        environment_index=1,
        generation_seed=42,
    )

    assert faulty.pair_id == healthy.pair_id
    assert faulty.environment == healthy.environment
    assert (
        faulty.topology_seed
        == healthy.topology_seed
    )
    assert (
        faulty.workload_seed
        == healthy.workload_seed
    )
    assert (
        faulty.observation_seed
        == healthy.observation_seed
    )

    assert (
        faulty.failure_type
        != FailureType.HEALTHY
    )
    assert (
        healthy.failure_type
        == FailureType.HEALTHY
    )


def test_counterfactual_generation_is_reproducible() -> None:
    first = create_counterfactual_plans(
        environment_index=1,
        generation_seed=42,
    )

    second = create_counterfactual_plans(
        environment_index=1,
        generation_seed=42,
    )

    assert first == second


def test_dataset_generates_two_scenarios_per_environment(
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

    assert len(manifest.records) == 8


def test_counterfactual_pair_remains_in_one_split(
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
        ood_environment_count=0,
        generation_seed=42,
    )

    pair_splits = {}

    for record in manifest.records:
        pair_splits.setdefault(
            record.pair_id,
            set(),
        ).add(record.split)

    assert all(
        len(splits) == 1
        for splits in pair_splits.values()
    )


def test_balanced_assignments_cover_every_class_in_core_splits() -> None:
    assignments = balanced_environment_assignments(20)

    for split in (
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
        DatasetSplit.TEST,
    ):
        classes = {
            failure_type
            for assigned_split, failure_type in assignments
            if assigned_split == split
        }
        assert classes == {
            FailureType.STORAGE_CAPACITY_SATURATION,
            FailureType.STORAGE_IO_DEGRADATION,
            FailureType.CLEANUP_JOB_FAILURE,
            FailureType.CATCHUP_WORKER_FAILURE,
        }


def test_hard_negative_cycle_covers_primary_false_positive_families() -> None:
    assert balanced_hard_negative_confounders(0) == [
        ConfounderType.TRANSIENT_RECORDING_ERRORS
    ]
    assert balanced_hard_negative_confounders(1) == [
        ConfounderType.HIGH_HEALTHY_WORKLOAD,
        ConfounderType.TRANSIENT_RECORDING_ERRORS,
    ]


def test_explicit_topology_groups_are_structurally_split_isolated(
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
        environment_count=0,
        ood_environment_count=0,
        generation_seed=606,
        topology_group_counts={
            DatasetSplit.TRAIN: 1,
            DatasetSplit.VALIDATION: 1,
        },
        dataset_version="0.4.0",
        simulator_version="0.5.0",
    )

    fingerprints_by_split = {
        split: {
            record.topology_fingerprint
            for record in manifest.records
            if record.split is split
        }
        for split in (
            DatasetSplit.TRAIN,
            DatasetSplit.VALIDATION,
        )
    }
    assert fingerprints_by_split[
        DatasetSplit.TRAIN
    ].isdisjoint(
        fingerprints_by_split[
            DatasetSplit.VALIDATION
        ]
    )
    assert all(
        record.fault_duration_hours is not None
        for record in manifest.records
        if not record.healthy
    )
