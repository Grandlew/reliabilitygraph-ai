from __future__ import annotations

import hashlib
import json
import random
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .models import FailureType


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD_TEST = "ood_test"


class MissingnessMode(str, Enum):
    NONE = "none"
    INDEPENDENT = "independent"
    SOURCE_SPECIFIC = "source_specific"
    CONTIGUOUS_OUTAGE = "contiguous_outage"


class ConfounderType(str, Enum):
    NONE = "none"
    HIGH_HEALTHY_WORKLOAD = "high_healthy_workload"
    RETENTION_INCREASE = "retention_increase"
    TEMPORARY_LATENCY_SPIKE = "temporary_latency_spike"
    HARMLESS_WORKER_RESTART = "harmless_worker_restart"
    TELEMETRY_COLLECTOR_OUTAGE = "telemetry_collector_outage"
    TRANSIENT_RECORDING_ERRORS = "transient_recording_errors"


class ScenarioEnvironment(BaseModel):
    environment_id: str = Field(min_length=1)

    room_count: int = Field(gt=0)
    floor_count: int = Field(gt=0)

    redundant_middleware: bool
    shared_storage: bool

    base_occupancy_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    evening_peak_multiplier: float = Field(ge=1.0)
    weekend_multiplier: float = Field(gt=0.0)

    catchup_recording_channels: int = Field(gt=0)
    average_bitrate_mbps: float = Field(gt=0.0)
    retention_days: int = Field(gt=0)

    scenario_duration_hours: int = Field(gt=0)
    sampling_interval_minutes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_environment(
        self,
    ) -> "ScenarioEnvironment":
        if self.floor_count > self.room_count:
            raise ValueError(
                "floor_count cannot exceed room_count."
            )

        if (
            self.scenario_duration_hours * 60
            < self.sampling_interval_minutes * 12
        ):
            raise ValueError(
                "Environment must support at least 12 samples."
            )

        return self


class MissingnessPlan(BaseModel):
    mode: MissingnessMode = MissingnessMode.NONE

    probability: float = Field(
        default=0.0,
        ge=0.0,
        le=0.80,
    )

    affected_signal_names: list[str] = Field(
        default_factory=list
    )

    outage_start_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    outage_duration_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=0.50,
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "MissingnessPlan":
        if self.mode == MissingnessMode.NONE:
            if self.probability != 0.0:
                raise ValueError(
                    "NONE missingness requires probability 0."
                )

        if self.mode == MissingnessMode.CONTIGUOUS_OUTAGE:
            if self.outage_start_fraction is None:
                raise ValueError(
                    "Contiguous outage requires a start fraction."
                )

            if self.outage_duration_fraction is None:
                raise ValueError(
                    "Contiguous outage requires a duration."
                )

        return self


class ScenarioPlan(BaseModel):
    scenario_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    environment: ScenarioEnvironment

    failure_type: FailureType
    fault_severity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    fault_injection_fraction: float | None = Field(
        default=None,
        ge=0.20,
        le=0.80,
    )

    confounders: list[ConfounderType] = Field(
        default_factory=list
    )
    missingness: MissingnessPlan = Field(
        default_factory=MissingnessPlan
    )

    topology_seed: int
    workload_seed: int
    observation_seed: int
    fault_seed: int

    intended_split: DatasetSplit | None = None

    @model_validator(mode="after")
    def validate_scenario_plan(self) -> "ScenarioPlan":
        is_healthy = self.failure_type == FailureType.HEALTHY

        if is_healthy:
            if self.fault_severity is not None:
                raise ValueError(
                    "Healthy scenarios cannot have fault severity."
                )

            if self.fault_injection_fraction is not None:
                raise ValueError(
                    "Healthy scenarios cannot have injection time."
                )

        else:
            if self.fault_severity is None:
                raise ValueError(
                    "Fault scenarios require severity."
                )

            if self.fault_injection_fraction is None:
                raise ValueError(
                    "Fault scenarios require injection fraction."
                )

        return self


def opaque_identifier(
    *,
    namespace: str,
    values: dict[str, Any],
) -> str:
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        f"{namespace}:{payload}".encode("utf-8")
    ).hexdigest()[:12]

    return f"{namespace}_{digest}"


def sample_environment(
    *,
    rng: random.Random,
    environment_index: int,
    ood: bool = False,
) -> ScenarioEnvironment:
    if ood:
        room_count = rng.choice([320, 400, 500])
        retention_days = rng.choice([18, 21])
        base_occupancy = rng.uniform(0.80, 0.98)
    else:
        room_count = rng.choice(
            [50, 80, 120, 180, 250]
        )
        retention_days = rng.choice(
            [2, 3, 5, 7, 10, 14]
        )
        base_occupancy = rng.uniform(0.35, 0.90)

    maximum_floors = min(
        10,
        max(2, room_count // 20),
    )

    floor_count = rng.randint(2, maximum_floors)

    values = {
        "index": environment_index,
        "room_count": room_count,
        "floor_count": floor_count,
        "retention_days": retention_days,
        "ood": ood,
    }

    return ScenarioEnvironment(
        environment_id=opaque_identifier(
            namespace="environment",
            values=values,
        ),
        room_count=room_count,
        floor_count=floor_count,
        redundant_middleware=rng.choice(
            [False, True]
        ),
        shared_storage=rng.choice([False, True]),
        base_occupancy_fraction=base_occupancy,
        evening_peak_multiplier=rng.uniform(
            1.05,
            1.60,
        ),
        weekend_multiplier=rng.uniform(
            0.90,
            1.25,
        ),
        catchup_recording_channels=rng.randint(
            10,
            70,
        ),
        average_bitrate_mbps=rng.uniform(
            2.5,
            8.0,
        ),
        retention_days=retention_days,
        scenario_duration_hours=rng.choice(
            [24, 36, 48, 72]
        ),
        sampling_interval_minutes=rng.choice(
            [5, 10, 15, 30]
        ),
    )


def sample_missingness_plan(
    *,
    rng: random.Random,
    stronger_ood: bool = False,
) -> MissingnessPlan:
    mode = rng.choices(
        population=[
            MissingnessMode.NONE,
            MissingnessMode.INDEPENDENT,
            MissingnessMode.SOURCE_SPECIFIC,
            MissingnessMode.CONTIGUOUS_OUTAGE,
        ],
        weights=[0.30, 0.35, 0.20, 0.15],
        k=1,
    )[0]

    if mode == MissingnessMode.NONE:
        return MissingnessPlan()

    maximum_probability = 0.35 if stronger_ood else 0.15

    if mode == MissingnessMode.INDEPENDENT:
        return MissingnessPlan(
            mode=mode,
            probability=rng.uniform(
                0.02,
                maximum_probability,
            ),
        )

    if mode == MissingnessMode.SOURCE_SPECIFIC:
        return MissingnessPlan(
            mode=mode,
            probability=rng.uniform(
                0.05,
                maximum_probability,
            ),
            affected_signal_names=[
                rng.choice(
                    [
                        "system.disk.io_latency",
                        "system.disk.io_errors",
                        "iptv.catchup.recording_failures",
                    ]
                )
            ],
        )

    return MissingnessPlan(
        mode=mode,
        probability=0.0,
        outage_start_fraction=rng.uniform(
            0.15,
            0.75,
        ),
        outage_duration_fraction=rng.uniform(
            0.03,
            0.15 if not stronger_ood else 0.30,
        ),
    )


def sample_confounders(
    *,
    rng: random.Random,
) -> list[ConfounderType]:
    available = [
        ConfounderType.HIGH_HEALTHY_WORKLOAD,
        ConfounderType.RETENTION_INCREASE,
        ConfounderType.TEMPORARY_LATENCY_SPIKE,
        ConfounderType.HARMLESS_WORKER_RESTART,
        ConfounderType.TRANSIENT_RECORDING_ERRORS,
    ]

    count = rng.choices(
        population=[0, 1, 2],
        weights=[0.35, 0.50, 0.15],
        k=1,
    )[0]

    if count == 0:
        return []

    return rng.sample(
        available,
        k=min(count, len(available)),
    )
