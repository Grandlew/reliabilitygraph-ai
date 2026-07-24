from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

from .dataset_builder import save_scenario
from .dataset_manifest import (
    DatasetManifest,
    ScenarioManifestRecord,
)
from .models import FailureType
from .scenario_design import (
    DatasetSplit,
    ScenarioPlan,
    opaque_identifier,
    sample_environment,
    sample_missingness_plan,
)
from .scenario_design import ConfounderType
from .scenario_factory import (
    build_planned_scenario,
)
from .split_manager import (
    deterministic_group_split,
    topology_fingerprint,
)
from .topology_generator import (
    generate_hotel_topology,
)


FAULT_TYPES = [
    FailureType.STORAGE_CAPACITY_SATURATION,
    FailureType.STORAGE_IO_DEGRADATION,
    FailureType.CLEANUP_JOB_FAILURE,
    FailureType.CATCHUP_WORKER_FAILURE,
]

V06_TOPOLOGY_GROUP_COUNTS = {
    DatasetSplit.TRAIN: 80,
    DatasetSplit.VALIDATION: 60,
    DatasetSplit.DEVELOPMENT_TEST: 30,
    DatasetSplit.LOCKED_TEST: 30,
    DatasetSplit.OOD_TEST: 30,
    DatasetSplit.SEMANTIC_CHALLENGE: 20,
}


def sample_fault_duration_hours(
    *,
    rng: random.Random,
    failure_type: FailureType,
    available_hours: float,
) -> float:
    """Sample balanced short, moderate, and long observable episodes."""
    if available_hours < 2.0:
        raise ValueError(
            "At least two post-injection hours are required"
        )
    duration_class = rng.choices(
        population=("short", "moderate", "long"),
        weights=(0.30, 0.40, 0.30),
        k=1,
    )[0]
    ranges = {
        "short": (
            3.0
            if failure_type
            == FailureType.STORAGE_CAPACITY_SATURATION
            else 2.0,
            5.0,
        ),
        "moderate": (5.0, 9.0),
        "long": (9.0, 18.0),
    }
    lower, upper = ranges[duration_class]
    upper = min(upper, available_hours)
    lower = min(lower, upper)
    return round(rng.uniform(lower, upper), 3)


def balanced_hard_negative_confounders(
    environment_index: int,
) -> list[ConfounderType]:
    """Cycle reproducible benign regimes through every topology cohort."""
    cohort_index = (
        environment_index // len(FAULT_TYPES)
    )
    position = environment_index % len(FAULT_TYPES)
    primary = [
        [ConfounderType.TRANSIENT_RECORDING_ERRORS],
        [
            ConfounderType.HIGH_HEALTHY_WORKLOAD,
            ConfounderType.TRANSIENT_RECORDING_ERRORS,
        ],
        [ConfounderType.TEMPORARY_LATENCY_SPIKE],
        [
            ConfounderType.HARMLESS_WORKER_RESTART,
            ConfounderType.RETENTION_INCREASE,
        ],
    ]
    secondary = [
        [],
        [ConfounderType.HIGH_HEALTHY_WORKLOAD],
        [ConfounderType.RETENTION_INCREASE],
        [
            ConfounderType.TEMPORARY_LATENCY_SPIKE,
            ConfounderType.TRANSIENT_RECORDING_ERRORS,
        ],
    ]
    return list(
        (primary if cohort_index % 2 == 0 else secondary)[
            position
        ]
    )


def create_counterfactual_plans(
    *,
    environment_index: int,
    generation_seed: int,
    ood: bool = False,
    failure_type: FailureType | None = None,
    intended_split: DatasetSplit | None = None,
    topology_group_index: int | None = None,
) -> tuple[
    ScenarioPlan,
    ScenarioPlan,
]:
    rng = random.Random(
        generation_seed * 100_000
        + environment_index
    )
    topology_group_index = (
        topology_group_index
        if topology_group_index is not None
        else environment_index
    )
    topology_rng = random.Random(
        generation_seed * 1_000_000
        + topology_group_index
    )

    environment = sample_environment(
        rng=topology_rng,
        environment_index=topology_group_index,
        ood=ood,
    )

    topology_seed = topology_rng.randint(
        0,
        2_147_483_647,
    )
    workload_seed = rng.randint(
        0,
        2_147_483_647,
    )
    observation_seed = rng.randint(
        0,
        2_147_483_647,
    )
    fault_seed = rng.randint(
        0,
        2_147_483_647,
    )

    selected_failure_type = (
        failure_type
        if failure_type is not None
        else rng.choice(FAULT_TYPES)
    )
    severity = rng.uniform(0.25, 1.0)
    injection_fraction = rng.uniform(
        0.25,
        0.65,
    )
    available_hours = max(
        2.0,
        environment.scenario_duration_hours
        * (1.0 - injection_fraction)
        - 2.0,
    )
    fault_duration_hours = sample_fault_duration_hours(
        rng=rng,
        failure_type=selected_failure_type,
        available_hours=available_hours,
    )

    confounders = balanced_hard_negative_confounders(
        environment_index
    )

    missingness = sample_missingness_plan(
        rng=rng,
        stronger_ood=ood,
    )

    pair_id = opaque_identifier(
        namespace="pair",
        values={
            "generation_seed": generation_seed,
            "environment_index": environment_index,
            "ood": ood,
        },
    )

    faulty_id = opaque_identifier(
        namespace="scenario",
        values={
            "pair_id": pair_id,
            "member": 0,
        },
    )

    healthy_id = opaque_identifier(
        namespace="scenario",
        values={
            "pair_id": pair_id,
            "member": 1,
        },
    )

    shared = {
        "pair_id": pair_id,
        "environment": environment,
        "confounders": confounders,
        "missingness": missingness,
        "topology_seed": topology_seed,
        "workload_seed": workload_seed,
        "observation_seed": observation_seed,
        "fault_seed": fault_seed,
        "intended_split": intended_split,
    }

    faulty = ScenarioPlan(
        scenario_id=faulty_id,
        failure_type=selected_failure_type,
        fault_severity=severity,
        fault_injection_fraction=(
            injection_fraction
        ),
        fault_duration_hours=fault_duration_hours,
        **shared,
    )

    healthy = ScenarioPlan(
        scenario_id=healthy_id,
        failure_type=FailureType.HEALTHY,
        fault_severity=None,
        fault_injection_fraction=None,
        fault_duration_hours=None,
        **shared,
    )

    return faulty, healthy


def balanced_environment_assignments(
    environment_count: int,
) -> list[tuple[DatasetSplit, FailureType]]:
    """Assign environments with complete, deterministic class coverage."""
    if environment_count <= 0:
        return []

    if environment_count >= 12:
        validation_count = max(
            4,
            round(environment_count * 0.20 / 4) * 4,
        )
        test_count = validation_count
        split_counts = {
            DatasetSplit.TRAIN: (
                environment_count
                - validation_count
                - test_count
            ),
            DatasetSplit.VALIDATION: validation_count,
            DatasetSplit.TEST: test_count,
        }
    else:
        split_order = [
            DatasetSplit.TRAIN,
            DatasetSplit.VALIDATION,
            DatasetSplit.TEST,
        ]
        split_counts = {
            split: sum(
                index % len(split_order) == split_index
                for index in range(environment_count)
            )
            for split_index, split in enumerate(split_order)
        }

    assignments: list[tuple[DatasetSplit, FailureType]] = []
    for split in (
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
        DatasetSplit.TEST,
    ):
        for index in range(split_counts[split]):
            assignments.append(
                (
                    split,
                    FAULT_TYPES[index % len(FAULT_TYPES)],
                )
            )

    return assignments


def generate_dataset(
    *,
    output_dir: Path,
    start_time: datetime,
    environment_count: int,
    ood_environment_count: int,
    generation_seed: int,
    topology_group_counts: (
        dict[DatasetSplit, int] | None
    ) = None,
    dataset_version: str = "0.3.0",
    simulator_version: str = "0.4.0",
) -> DatasetManifest:
    if environment_count <= 0 and topology_group_counts is None:
        raise ValueError(
            "environment_count must be positive."
        )

    if ood_environment_count < 0:
        raise ValueError(
            "ood_environment_count cannot be negative."
        )

    records: list[ScenarioManifestRecord] = []

    explicit_group_specs = topology_group_counts is not None
    if explicit_group_specs:
        invalid_counts = {
            split: count
            for split, count in topology_group_counts.items()
            if count < 0
        }
        if invalid_counts:
            raise ValueError(
                "Topology group counts cannot be negative"
            )
        raw_environment_specs = []
        environment_index = 0
        split_order = (
            DatasetSplit.TRAIN,
            DatasetSplit.VALIDATION,
            DatasetSplit.DEVELOPMENT_TEST,
            DatasetSplit.LOCKED_TEST,
            DatasetSplit.OOD_TEST,
            DatasetSplit.SEMANTIC_CHALLENGE,
            DatasetSplit.TEST,
        )
        for split in split_order:
            for group_index in range(
                topology_group_counts.get(split, 0)
            ):
                for failure_type in FAULT_TYPES:
                    raw_environment_specs.append(
                        (
                            environment_index,
                            split == DatasetSplit.OOD_TEST,
                            split,
                            failure_type,
                            group_index,
                        )
                    )
                    environment_index += 1
    else:
        raw_environment_specs = [
            (index, False, split, failure_type)
            for index, (split, failure_type) in enumerate(
                balanced_environment_assignments(
                    environment_count
                )
            )
        ]

        raw_environment_specs.extend(
            [
                (
                    environment_count + index,
                    True,
                    DatasetSplit.OOD_TEST,
                    FAULT_TYPES[index % len(FAULT_TYPES)],
                )
                for index in range(
                    ood_environment_count
                )
            ]
        )

    split_sizes: dict[tuple[bool, DatasetSplit], int] = {}
    for raw_spec in raw_environment_specs:
        _, ood, split, _ = raw_spec[:4]
        key = (ood, split)
        split_sizes[key] = split_sizes.get(key, 0) + 1

    split_positions: dict[tuple[bool, DatasetSplit], int] = {}
    split_codes = {
        DatasetSplit.TRAIN: 1,
        DatasetSplit.VALIDATION: 2,
        DatasetSplit.TEST: 3,
        DatasetSplit.OOD_TEST: 4,
        DatasetSplit.DEVELOPMENT_TEST: 5,
        DatasetSplit.LOCKED_TEST: 6,
        DatasetSplit.SEMANTIC_CHALLENGE: 7,
    }
    environment_specs = []
    for raw_spec in raw_environment_specs:
        (
            environment_index,
            ood,
            intended_split,
            failure_type,
        ) = raw_spec[:4]
        key = (ood, intended_split)
        position = split_positions.get(key, 0)
        split_positions[key] = position + 1
        if explicit_group_specs:
            topology_group_index = (
                split_codes[intended_split] * 10_000
                + int(raw_spec[4])
            )
        else:
            complete_cohorts = (
                split_sizes[key] // len(FAULT_TYPES)
            )
            if position < complete_cohorts * len(FAULT_TYPES):
                topology_group_index = (
                    split_codes[intended_split] * 10_000
                    + position // len(FAULT_TYPES)
                )
            else:
                # In reduced test datasets an incomplete cohort must not
                # share a topology across distinct fault classes or splits.
                topology_group_index = (
                    split_codes[intended_split] * 10_000
                    + 5_000
                    + position
                )
        environment_specs.append(
            (
                environment_index,
                ood,
                intended_split,
                failure_type,
                topology_group_index,
            )
        )

    resolved_group_indices: dict[
        tuple[DatasetSplit, int],
        int,
    ] = {}
    fingerprint_owners: dict[
        str,
        tuple[DatasetSplit, int],
    ] = {}

    for (
        environment_index,
        ood,
        intended_split,
        failure_type,
        topology_group_index,
    ) in environment_specs:
        if explicit_group_specs:
            group_key = (
                intended_split,
                topology_group_index,
            )
            if group_key not in resolved_group_indices:
                for attempt in range(1_000):
                    candidate_group_index = (
                        topology_group_index
                        + attempt * 1_000_000
                    )
                    preview_faulty, _ = (
                        create_counterfactual_plans(
                            environment_index=environment_index,
                            generation_seed=generation_seed,
                            ood=ood,
                            failure_type=failure_type,
                            intended_split=intended_split,
                            topology_group_index=(
                                candidate_group_index
                            ),
                        )
                    )
                    preview_topology = (
                        generate_hotel_topology(
                            room_count=(
                                preview_faulty.environment.room_count
                            ),
                            floor_count=(
                                preview_faulty.environment.floor_count
                            ),
                            redundant_middleware=(
                                preview_faulty.environment.redundant_middleware
                            ),
                            shared_storage=(
                                preview_faulty.environment.shared_storage
                            ),
                            seed=(
                                preview_faulty.topology_seed
                            ),
                        )
                    )
                    preview_fingerprint = (
                        topology_fingerprint(
                            preview_topology.model_dump(
                                mode="json"
                            )
                        )
                    )
                    if (
                        preview_fingerprint
                        not in fingerprint_owners
                    ):
                        resolved_group_indices[
                            group_key
                        ] = candidate_group_index
                        fingerprint_owners[
                            preview_fingerprint
                        ] = group_key
                        break
                else:
                    raise RuntimeError(
                        "Unable to sample a unique structural "
                        f"topology for {group_key}"
                    )
            topology_group_index = (
                resolved_group_indices[group_key]
            )
        faulty_plan, healthy_plan = (
            create_counterfactual_plans(
                environment_index=environment_index,
                generation_seed=generation_seed,
                ood=ood,
                failure_type=failure_type,
                intended_split=intended_split,
                topology_group_index=topology_group_index,
            )
        )

        built_pair = []

        for plan in (
            faulty_plan,
            healthy_plan,
        ):
            scenario, missingness_report = (
                build_planned_scenario(
                    plan=plan,
                    start_time=start_time,
                )
            )

            topology = scenario.observable[
                "topology"
            ]

            fingerprint = topology_fingerprint(
                topology
            )

            split = (
                plan.intended_split
                or deterministic_group_split(
                    group_id=fingerprint,
                    ood=ood,
                )
            )

            scenario_dir = output_dir / split.value

            save_scenario(
                scenario=scenario,
                output_dir=scenario_dir,
                scenario_id=plan.scenario_id,
            )

            observable_path = (
                scenario_dir
                / f"{plan.scenario_id}.observable.json"
            )

            hidden_path = (
                scenario_dir
                / f"{plan.scenario_id}.hidden.json"
            )

            record = ScenarioManifestRecord(
                scenario_id=plan.scenario_id,
                pair_id=plan.pair_id,
                environment_id=(
                    plan.environment.environment_id
                ),
                topology_fingerprint=fingerprint,
                split=split,
                failure_type=(
                    plan.failure_type.value
                ),
                healthy=(
                    plan.failure_type
                    == FailureType.HEALTHY
                ),
                room_count=(
                    plan.environment.room_count
                ),
                floor_count=(
                    plan.environment.floor_count
                ),
                retention_days=(
                    plan.environment.retention_days
                ),
                base_occupancy_fraction=(
                    plan.environment.base_occupancy_fraction
                ),
                catchup_recording_channels=(
                    plan.environment.catchup_recording_channels
                ),
                average_bitrate_mbps=(
                    plan.environment.average_bitrate_mbps
                ),
                shared_storage=(
                    plan.environment.shared_storage
                ),
                redundant_middleware=(
                    plan.environment.redundant_middleware
                ),
                fault_severity=(
                    plan.fault_severity
                ),
                fault_injection_fraction=(
                    plan.fault_injection_fraction
                ),
                fault_duration_hours=(
                    plan.fault_duration_hours
                ),
                confounders=[
                    item.value
                    for item in plan.confounders
                ],
                missingness_mode=(
                    plan.missingness.mode.value
                ),
                missing_fraction=float(
                    missingness_report[
                        "removed_fraction"
                    ]
                ),
                observable_path=str(
                    observable_path
                ),
                hidden_path=str(hidden_path),
                topology_seed=(
                    plan.topology_seed
                ),
                workload_seed=(
                    plan.workload_seed
                ),
                observation_seed=(
                    plan.observation_seed
                ),
                fault_seed=plan.fault_seed,
            )

            records.append(record)
            built_pair.append(record)

        pair_splits = {
            record.split
            for record in built_pair
        }

        if len(pair_splits) != 1:
            raise RuntimeError(
                "Counterfactual pair was split "
                "across dataset partitions."
            )

    return DatasetManifest(
        dataset_name=(
            "nrim_domain_randomized_reliability"
        ),
        dataset_version=dataset_version,
        simulator_version=simulator_version,
        generation_seed=generation_seed,
        records=records,
    )
