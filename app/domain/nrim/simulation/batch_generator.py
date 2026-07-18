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
    sample_confounders,
    sample_environment,
    sample_missingness_plan,
)
from .scenario_factory import (
    build_planned_scenario,
)
from .split_manager import (
    deterministic_group_split,
    topology_fingerprint,
)


FAULT_TYPES = [
    FailureType.STORAGE_CAPACITY_SATURATION,
    FailureType.STORAGE_IO_DEGRADATION,
    FailureType.CLEANUP_JOB_FAILURE,
    FailureType.CATCHUP_WORKER_FAILURE,
]


def create_counterfactual_plans(
    *,
    environment_index: int,
    generation_seed: int,
    ood: bool = False,
) -> tuple[
    ScenarioPlan,
    ScenarioPlan,
]:
    rng = random.Random(
        generation_seed * 100_000
        + environment_index
    )

    environment = sample_environment(
        rng=rng,
        environment_index=environment_index,
        ood=ood,
    )

    topology_seed = rng.randint(
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

    failure_type = rng.choice(FAULT_TYPES)
    severity = rng.uniform(0.25, 1.0)
    injection_fraction = rng.uniform(
        0.25,
        0.70,
    )

    confounders = sample_confounders(rng=rng)

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
    }

    faulty = ScenarioPlan(
        scenario_id=faulty_id,
        failure_type=failure_type,
        fault_severity=severity,
        fault_injection_fraction=(
            injection_fraction
        ),
        **shared,
    )

    healthy = ScenarioPlan(
        scenario_id=healthy_id,
        failure_type=FailureType.HEALTHY,
        fault_severity=None,
        fault_injection_fraction=None,
        **shared,
    )

    return faulty, healthy


def generate_dataset(
    *,
    output_dir: Path,
    start_time: datetime,
    environment_count: int,
    ood_environment_count: int,
    generation_seed: int,
) -> DatasetManifest:
    if environment_count <= 0:
        raise ValueError(
            "environment_count must be positive."
        )

    if ood_environment_count < 0:
        raise ValueError(
            "ood_environment_count cannot be negative."
        )

    records: list[ScenarioManifestRecord] = []

    environment_specs = [
        (index, False)
        for index in range(environment_count)
    ]

    environment_specs.extend(
        [
            (
                environment_count + index,
                True,
            )
            for index in range(
                ood_environment_count
            )
        ]
    )

    for environment_index, ood in environment_specs:
        faulty_plan, healthy_plan = (
            create_counterfactual_plans(
                environment_index=environment_index,
                generation_seed=generation_seed,
                ood=ood,
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

            split = deterministic_group_split(
                group_id=fingerprint,
                ood=ood,
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
                fault_severity=(
                    plan.fault_severity
                ),
                fault_injection_fraction=(
                    plan.fault_injection_fraction
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
        dataset_version="0.1.0",
        simulator_version="0.2.0",
        generation_seed=generation_seed,
        records=records,
    )
