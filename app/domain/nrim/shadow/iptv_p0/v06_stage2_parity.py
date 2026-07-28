from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.domain.nrim.benchmark.dataset_loader import (
    load_model_ready_manifest,
    resolve_window_record_path,
)
from app.domain.nrim.simulation.locked_test_governance import (
    resolve_manifest_record_path,
    resolve_seal_manifest_path,
    verify_locked_test_seal,
)

from ..hashing import canonical_hash, file_hash
from ..parity import (
    normalized_reference_window,
    synthetic_shadow_snapshot,
)
from ..privacy import Pseudonymizer
from ..replay import ServingFeatureBuilder
from .contracts import utc


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class V06Stage2ExampleSeal(StrictModel):
    schema_version: str = Field(pattern=r"^0\.8\.[0-9]+$")
    example_id: str = Field(min_length=8, max_length=128)
    scenario_id: str = Field(pattern=r"^scenario_[0-9a-f]{12}$")
    window_id: str = Field(pattern=r"^window_[0-9a-f]{16}$")
    source_locked_seal_path: str
    source_locked_seal_commitment_sha256: str = Field(
        pattern=SHA256_PATTERN
    )
    model_ready_manifest_path: str
    model_ready_manifest_canonical_sha256: str = Field(
        pattern=SHA256_PATTERN
    )
    reference_window_path: str
    reference_window_canonical_sha256: str = Field(
        pattern=SHA256_PATTERN
    )
    event_cutoff_utc: datetime
    knowledge_cutoff_utc: datetime
    candidate_count: int = Field(gt=1)
    feature_count: int = Field(gt=1)
    applicability_mask_count: int = Field(gt=0)
    missingness_mask_count: int = Field(gt=0)
    declared_absolute_tolerance: float = Field(ge=0.0, le=1e-6)
    commitment_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator(
        "source_locked_seal_path",
        "model_ready_manifest_path",
        "reference_window_path",
    )
    @classmethod
    def validate_repository_path(cls, value: str) -> str:
        portable = PurePosixPath(value)
        if (
            portable.is_absolute()
            or ".." in portable.parts
            or "\\" in value
            or not portable.parts
        ):
            raise ValueError(
                "Sealed Stage 2 paths must be repository-relative POSIX paths"
            )
        return value

    @field_validator("event_cutoff_utc", "knowledge_cutoff_utc")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> Self:
        if self.knowledge_cutoff_utc < self.event_cutoff_utc:
            raise ValueError(
                "Knowledge cutoff cannot precede the event cutoff"
            )
        return self

    @property
    def computed_commitment_sha256(self) -> str:
        return canonical_hash(
            self.model_dump(
                mode="json",
                exclude={"commitment_sha256"},
            )
        )


@dataclass(frozen=True)
class V06Stage2Example:
    seal: V06Stage2ExampleSeal
    scenario_metadata: dict[str, Any]
    observable_scenario: dict[str, Any]
    reference_window: dict[str, Any]


def build_v06_stage2_example_seal() -> V06Stage2ExampleSeal:
    provisional = V06Stage2ExampleSeal(
        schema_version="0.8.0",
        example_id="v06_locked_stage2_window_68fe47e9b82a3632",
        scenario_id="scenario_0004e9f6aafd",
        window_id="window_68fe47e9b82a3632",
        source_locked_seal_path=(
            "app/domain/nrim/examples/simulation/day08_dataset_v06/"
            "governance/locked_test_seal.json"
        ),
        source_locked_seal_commitment_sha256=(
            "b0a5b115a4a50282cc3f1726855a82c5b4771d10be8c5f245fce29e192478cea"
        ),
        model_ready_manifest_path=(
            "app/domain/nrim/examples/simulation/day09_model_ready_v06/"
            "manifest.json"
        ),
        model_ready_manifest_canonical_sha256=(
            "8dbde22820837f6ae8c60fce7f428d5e8795f517e007cac387a936f093f9b919"
        ),
        reference_window_path=(
            "app/domain/nrim/examples/simulation/day09_model_ready_v06/"
            "locked_test/window_68fe47e9b82a3632.json"
        ),
        reference_window_canonical_sha256=(
            "2496a5ca9720d355363c1e05dfa0205c525a78937a709e7df0afecaa9e3dcc08"
        ),
        event_cutoff_utc=datetime.fromisoformat(
            "2026-07-18T06:00:00+00:00"
        ),
        knowledge_cutoff_utc=datetime.fromisoformat(
            "2026-07-18T06:00:00+00:00"
        ),
        candidate_count=29,
        feature_count=114,
        applicability_mask_count=7,
        missingness_mask_count=7,
        declared_absolute_tolerance=1e-12,
        commitment_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "commitment_sha256": provisional.computed_commitment_sha256
        }
    )


def _repository_path(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    portable = PurePosixPath(relative)
    path = root.joinpath(*portable.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Sealed Stage 2 path escapes the repository")
    return path


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Sealed {label} is missing: {path}")
    if file_hash(path) != expected:
        raise ValueError(f"Sealed {label} hash differs")


def _require_json_commitment(
    path: Path,
    expected: str,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Sealed {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_hash(value) != expected:
        raise ValueError(f"Sealed {label} canonical commitment differs")
    return value


def load_sealed_v06_stage2_example(
    *,
    seal_path: Path,
    repository_root: Path,
) -> V06Stage2Example:
    seal = V06Stage2ExampleSeal.model_validate_json(
        seal_path.read_text(encoding="utf-8")
    )
    if seal.computed_commitment_sha256 != seal.commitment_sha256:
        raise ValueError("Stage 2 example seal commitment differs")

    source_seal_path = _repository_path(
        repository_root,
        seal.source_locked_seal_path,
    )
    source_seal = verify_locked_test_seal(seal_path=source_seal_path)
    if (
        source_seal.get("commitment_sha256")
        != seal.source_locked_seal_commitment_sha256
    ):
        raise ValueError("Source locked-test seal commitment differs")
    locked_manifest_path = resolve_seal_manifest_path(
        seal_path=source_seal_path,
        seal=source_seal,
        field="locked_manifest_path",
    )
    locked_manifest = json.loads(
        locked_manifest_path.read_text(encoding="utf-8")
    )
    try:
        scenario_metadata = next(
            item
            for item in locked_manifest["records"]
            if item["scenario_id"] == seal.scenario_id
        )
    except StopIteration as error:
        raise ValueError(
            "Sealed Stage 2 scenario is not in the locked manifest"
        ) from error
    observable_path = resolve_manifest_record_path(
        manifest_path=locked_manifest_path,
        record=scenario_metadata,
        kind="observable",
    )
    expected_observable_hash = source_seal["file_hashes"][
        f"{seal.scenario_id}:observable"
    ]
    _require_hash(
        observable_path,
        expected_observable_hash,
        "source observable",
    )

    model_manifest_path = _repository_path(
        repository_root,
        seal.model_ready_manifest_path,
    )
    _require_json_commitment(
        model_manifest_path,
        seal.model_ready_manifest_canonical_sha256,
        "model-ready manifest",
    )
    model_manifest = load_model_ready_manifest(model_manifest_path)
    try:
        window_record = next(
            item
            for item in model_manifest["records"]
            if item["window_id"] == seal.window_id
            and item["split"] == "locked_test"
        )
    except StopIteration as error:
        raise ValueError(
            "Sealed Stage 2 window is not in the locked-test split"
        ) from error
    resolved_window_path = resolve_window_record_path(
        manifest=model_manifest,
        record=window_record,
    )
    declared_window_path = _repository_path(
        repository_root,
        seal.reference_window_path,
    )
    if resolved_window_path.resolve() != declared_window_path:
        raise ValueError(
            "Sealed Stage 2 window path differs from its manifest"
        )
    reference_window = _require_json_commitment(
        declared_window_path,
        seal.reference_window_canonical_sha256,
        "Stage 2 reference window",
    )
    if reference_window["source_scenario_id"] != seal.scenario_id:
        raise ValueError("Stage 2 window and source scenario differ")
    if reference_window["window_id"] != seal.window_id:
        raise ValueError("Stage 2 window identity differs")
    if utc(datetime.fromisoformat(reference_window["observation_cutoff"])) != (
        seal.event_cutoff_utc
    ):
        raise ValueError("Stage 2 event cutoff differs from its seal")
    if len(reference_window["node_ids"]) != seal.candidate_count:
        raise ValueError("Stage 2 candidate count differs from its seal")
    if len(reference_window["node_feature_names"]) != seal.feature_count:
        raise ValueError("Stage 2 feature count differs from its seal")
    applicability_count = sum(
        name.endswith("__applicable")
        for name in reference_window["node_feature_names"]
    )
    missingness_count = sum(
        name.endswith("__missing")
        for name in reference_window["node_feature_names"]
    )
    if applicability_count != seal.applicability_mask_count:
        raise ValueError("Stage 2 applicability-mask count differs")
    if missingness_count != seal.missingness_mask_count:
        raise ValueError("Stage 2 missingness-mask count differs")

    return V06Stage2Example(
        seal=seal,
        scenario_metadata=scenario_metadata,
        observable_scenario=json.loads(
            observable_path.read_text(encoding="utf-8")
        ),
        reference_window=reference_window,
    )


def reconstruct_iptv_p0_v06_stage2(
    *,
    observable_scenario: dict[str, Any],
    scenario_metadata: dict[str, Any],
    reference_window: dict[str, Any],
    knowledge_cutoff_utc: datetime,
    pseudonymization_secret: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the frozen v0.6 Stage 2 tensor without inference."""

    pseudonymizer = Pseudonymizer(
        pseudonymization_secret,
        key_id="iptv-p0-v06-stage2-parity",
    )
    snapshot, node_map = synthetic_shadow_snapshot(
        observable_scenario=observable_scenario,
        scenario_metadata=scenario_metadata,
        reference_window=reference_window,
        pseudonymizer=pseudonymizer,
        knowledge_cutoff_utc=knowledge_cutoff_utc,
    )
    actual = ServingFeatureBuilder().build_sequence([snapshot])[0]
    expected = normalized_reference_window(
        reference_window=reference_window,
        node_map=node_map,
    )
    return actual, expected


def compare_v06_stage2(
    *,
    actual: dict[str, Any],
    expected: dict[str, Any],
    absolute_tolerance: float,
) -> dict[str, Any]:
    if (
        not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
    ):
        raise ValueError("Stage 2 tolerance must be finite and non-negative")

    actual_candidates = list(actual.get("node_ids", ()))
    expected_candidates = list(expected.get("node_ids", ()))
    actual_names = list(actual.get("node_feature_names", ()))
    expected_names = list(expected.get("node_feature_names", ()))
    actual_rows = list(actual.get("node_features", ()))
    expected_rows = list(expected.get("node_features", ()))
    actual_shape = (
        len(actual_rows),
        len(actual_rows[0]) if actual_rows else 0,
    )
    expected_shape = (
        len(expected_rows),
        len(expected_rows[0]) if expected_rows else 0,
    )
    rectangular_actual = all(
        len(row) == actual_shape[1] for row in actual_rows
    )
    rectangular_expected = all(
        len(row) == expected_shape[1] for row in expected_rows
    )
    candidate_order_matches = actual_candidates == expected_candidates
    feature_names_match = actual_names == expected_names
    tensor_shape_matches = (
        rectangular_actual
        and rectangular_expected
        and actual_shape == expected_shape
        and actual_shape
        == (len(actual_candidates), len(actual_names))
    )

    applicability_mask_matches = False
    missingness_mask_matches = False
    values_within_tolerance = False
    max_absolute_error = 0.0
    numerical_mismatch_count = 0
    first_differences: list[dict[str, Any]] = []
    if feature_names_match and tensor_shape_matches:
        applicability_indices = tuple(
            index
            for index, name in enumerate(expected_names)
            if name.endswith("__applicable")
        )
        missingness_indices = tuple(
            index
            for index, name in enumerate(expected_names)
            if name.endswith("__missing")
        )
        applicability_mask_matches = all(
            actual_rows[row][column] == expected_rows[row][column]
            for row in range(expected_shape[0])
            for column in applicability_indices
        )
        missingness_mask_matches = all(
            actual_rows[row][column] == expected_rows[row][column]
            for row in range(expected_shape[0])
            for column in missingness_indices
        )
        mask_indices = set(applicability_indices) | set(missingness_indices)
        for row in range(expected_shape[0]):
            for column in range(expected_shape[1]):
                if column in mask_indices:
                    continue
                actual_value = float(actual_rows[row][column])
                expected_value = float(expected_rows[row][column])
                if not (
                    math.isfinite(actual_value)
                    and math.isfinite(expected_value)
                ):
                    difference = math.inf
                else:
                    difference = abs(actual_value - expected_value)
                max_absolute_error = max(max_absolute_error, difference)
                if difference > absolute_tolerance:
                    numerical_mismatch_count += 1
                    if len(first_differences) < 10:
                        first_differences.append(
                            {
                                "candidate_index": row,
                                "candidate": expected_candidates[row],
                                "feature_index": column,
                                "feature_name": expected_names[column],
                                "actual": actual_value,
                                "expected": expected_value,
                                "absolute_error": difference,
                            }
                        )
        values_within_tolerance = numerical_mismatch_count == 0

    expected_has_distinct_rows = len(
        {tuple(row) for row in expected_rows}
    ) > 1
    actual_is_broadcast = (
        len(actual_rows) > 1
        and len({tuple(row) for row in actual_rows}) == 1
    )
    broadcast_detected = expected_has_distinct_rows and actual_is_broadcast
    matches = all(
        (
            candidate_order_matches,
            feature_names_match,
            tensor_shape_matches,
            applicability_mask_matches,
            missingness_mask_matches,
            values_within_tolerance,
            not broadcast_detected,
        )
    )
    return {
        "matches": matches,
        "candidate_order_matches": candidate_order_matches,
        "feature_names_match": feature_names_match,
        "tensor_shape_matches": tensor_shape_matches,
        "actual_shape": actual_shape,
        "expected_shape": expected_shape,
        "applicability_mask_matches": applicability_mask_matches,
        "missingness_mask_matches": missingness_mask_matches,
        "values_within_tolerance": values_within_tolerance,
        "declared_absolute_tolerance": absolute_tolerance,
        "max_absolute_error": max_absolute_error,
        "numerical_mismatch_count": numerical_mismatch_count,
        "broadcast_detected": broadcast_detected,
        "first_differences": tuple(first_differences),
    }


def verify_sealed_v06_stage2_parity(
    *,
    seal_path: Path,
    repository_root: Path,
    pseudonymization_secret: bytes,
) -> dict[str, Any]:
    example = load_sealed_v06_stage2_example(
        seal_path=seal_path,
        repository_root=repository_root,
    )
    actual, expected = reconstruct_iptv_p0_v06_stage2(
        observable_scenario=example.observable_scenario,
        scenario_metadata=example.scenario_metadata,
        reference_window=example.reference_window,
        knowledge_cutoff_utc=example.seal.knowledge_cutoff_utc,
        pseudonymization_secret=pseudonymization_secret,
    )
    comparison = compare_v06_stage2(
        actual=actual,
        expected=expected,
        absolute_tolerance=example.seal.declared_absolute_tolerance,
    )
    return {
        **comparison,
        "example_id": example.seal.example_id,
        "scenario_id": example.seal.scenario_id,
        "window_id": example.seal.window_id,
        "reference_window_canonical_sha256": (
            example.seal.reference_window_canonical_sha256
        ),
        "actual_stage2_sha256": canonical_hash(actual),
        "expected_stage2_sha256": canonical_hash(expected),
    }
