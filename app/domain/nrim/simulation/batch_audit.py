from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .dataset_manifest import DatasetManifest
from .quality_audit import (
    audit_observable_scenario,
    audit_root_cause_labels,
)
from .split_manager import validate_group_isolation


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def audit_manifest(
    manifest: DatasetManifest,
) -> dict[str, Any]:
    records = [
        record.model_dump(mode="json")
        for record in manifest.records
    ]

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(
        validate_group_isolation(records)
    )

    failure_counts = Counter(
        record["failure_type"]
        for record in records
        if not record["healthy"]
    )

    split_counts = Counter(
        record["split"]
        for record in records
    )

    topology_by_split: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in records:
        topology_by_split[
            record["split"]
        ].add(record["topology_fingerprint"])

    missingness_by_class: dict[
        str,
        list[float],
    ] = defaultdict(list)

    for record in records:
        missingness_by_class[
            record["failure_type"]
        ].append(
            float(record["missing_fraction"])
        )

    mean_missingness = {
        failure_type: round(
            mean(values),
            6,
        )
        for failure_type, values
        in missingness_by_class.items()
        if values
    }

    if failure_counts:
        maximum_count = max(
            failure_counts.values()
        )
        minimum_count = min(
            failure_counts.values()
        )

        if minimum_count == 0:
            warnings.append(
                "At least one failure class is absent."
            )

        elif maximum_count / minimum_count > 2.0:
            warnings.append(
                "Failure-class imbalance exceeds 2:1."
            )

    for record in records:
        observable = load_json(
            record["observable_path"]
        )

        leakage = audit_observable_scenario(
            observable
        )
        integrity = audit_root_cause_labels(
            observable
        )

        warnings.extend(
            [
                f"{record['scenario_id']}: {item}"
                for item in leakage
            ]
        )

        errors.extend(
            [
                f"{record['scenario_id']}: {item}"
                for item in integrity
            ]
        )

    return {
        "scenario_count": len(records),
        "split_counts": dict(split_counts),
        "fault_class_counts": dict(
            failure_counts
        ),
        "unique_topologies_by_split": {
            split: len(values)
            for split, values
            in topology_by_split.items()
        },
        "mean_missingness_by_class": (
            mean_missingness
        ),
        "errors": errors,
        "warnings": warnings,
    }


def pre_fault_pair_signature(
    observable: dict[str, Any],
    *,
    fault_injection_time: str,
) -> list[tuple[str, str, float | int]]:
    injection_time = fault_injection_time

    signature: list[
        tuple[str, str, float | int]
    ] = []

    for event in observable.get(
        "telemetry",
        [],
    ):
        observed_at = str(
            event.get("observed_at")
        )

        if observed_at >= injection_time:
            continue

        value = event.get("value")

        if isinstance(value, (int, float)):
            signature.append(
                (
                    observed_at,
                    str(event.get("signal_name")),
                    value,
                )
            )

    return sorted(signature)
