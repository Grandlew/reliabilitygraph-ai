from datetime import datetime, timezone

from app.domain.nrim.simulation.batch_generator import (
    create_counterfactual_plans,
    generate_dataset,
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
