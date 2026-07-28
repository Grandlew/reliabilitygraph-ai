from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from app.domain.nrim.shadow.iptv_p0 import feature_reconstruction
from app.domain.nrim.shadow.iptv_p0.v06_stage2_parity import (
    build_sealed_v06_stage2_reference,
    compare_v06_stage2,
    load_sealed_v06_stage2_example,
    reconstruct_iptv_p0_v06_stage2,
    reconstruct_legacy_v06_serving_stage2,
    verify_sealed_v06_stage2_parity,
)


ROOT = Path(__file__).resolve().parents[5]
SEAL_PATH = (
    ROOT
    / "app/domain/nrim/examples/shadow/iptv_p0_v0_8"
    / "parity/v06_stage2_example.seal.json"
)
SECRET = b"iptv-p0-v06-parity-" + b"x" * 32


@pytest.fixture(scope="module")
def sealed_example():
    return load_sealed_v06_stage2_example(
        seal_path=SEAL_PATH,
        repository_root=ROOT,
    )


@pytest.fixture(scope="module")
def parity_windows(sealed_example):
    actual = reconstruct_iptv_p0_v06_stage2(
        observable_scenario=sealed_example.observable_scenario,
        observation_start_utc=sealed_example.seal.observation_start_utc,
        event_cutoff_utc=sealed_example.seal.event_cutoff_utc,
        knowledge_cutoff_utc=(
            sealed_example.seal.knowledge_cutoff_utc
        ),
        pseudonymization_secret=SECRET,
    )
    expected = build_sealed_v06_stage2_reference(
        reference_window=sealed_example.reference_window,
        pseudonymization_secret=SECRET,
    )
    return sealed_example, actual, expected


def test_legacy_v06_serving_builder_is_self_consistent_with_export(
    sealed_example,
):
    actual, expected = reconstruct_legacy_v06_serving_stage2(
        observable_scenario=sealed_example.observable_scenario,
        scenario_metadata=sealed_example.scenario_metadata,
        reference_window=sealed_example.reference_window,
        knowledge_cutoff_utc=(
            sealed_example.seal.knowledge_cutoff_utc
        ),
        pseudonymization_secret=SECRET,
    )

    result = compare_v06_stage2(
        actual=actual,
        expected=expected,
        absolute_tolerance=(
            sealed_example.seal.declared_absolute_tolerance
        ),
    )
    assert result["matches"], result["first_differences"]


def test_iptv_p0_end_to_end_exactly_reconstructs_sealed_v06_stage2():
    result = verify_sealed_v06_stage2_parity(
        seal_path=SEAL_PATH,
        repository_root=ROOT,
        pseudonymization_secret=SECRET,
    )

    assert result["matches"], result["first_differences"]
    assert result["candidate_order_matches"]
    assert result["feature_names_match"]
    assert result["tensor_shape_matches"]
    assert result["expected_shape"] == (29, 114)
    assert result["applicability_mask_matches"]
    assert result["missingness_mask_matches"]
    assert result["values_within_tolerance"]
    assert result["max_absolute_error"] <= 1e-12
    assert not result["broadcast_detected"]
    assert (
        result["actual_stage2_sha256"]
        == result["expected_stage2_sha256"]
    )


def test_end_to_end_parity_depends_on_iptv_feature_reconstruction(
    sealed_example,
    monkeypatch,
):
    original = feature_reconstruction.reconstruct_frozen_features

    def break_local_reconstruction(**kwargs):
        result = original(**kwargs)
        values = list(result.stage2.values)
        index = next(
            index for index, value in enumerate(values) if value is not None
        )
        values[index] = float(values[index]) + 1.0
        return result.model_copy(
            update={
                "stage2": result.stage2.model_copy(
                    update={"values": tuple(values)}
                )
            }
        )

    monkeypatch.setattr(
        feature_reconstruction,
        "reconstruct_frozen_features",
        break_local_reconstruction,
    )

    result = verify_sealed_v06_stage2_parity(
        seal_path=SEAL_PATH,
        repository_root=ROOT,
        pseudonymization_secret=SECRET,
    )
    assert not result["matches"]
    assert not result["values_within_tolerance"]


def test_actual_stage2_is_derived_from_adapter_records(
    sealed_example,
):
    changed_observable = deepcopy(sealed_example.observable_scenario)
    changed_event = next(
        event
        for event in changed_observable["telemetry"]
        if event["value"] is not None
        and event["observed_at"] < "2026-07-18T06:00:00Z"
    )
    changed_event["value"] = float(changed_event["value"]) + 10.0

    actual = reconstruct_iptv_p0_v06_stage2(
        observable_scenario=changed_observable,
        observation_start_utc=sealed_example.seal.observation_start_utc,
        event_cutoff_utc=sealed_example.seal.event_cutoff_utc,
        knowledge_cutoff_utc=(
            sealed_example.seal.knowledge_cutoff_utc
        ),
        pseudonymization_secret=SECRET,
    )
    expected = build_sealed_v06_stage2_reference(
        reference_window=sealed_example.reference_window,
        pseudonymization_secret=SECRET,
    )
    result = compare_v06_stage2(
        actual=actual,
        expected=expected,
        absolute_tolerance=(
            sealed_example.seal.declared_absolute_tolerance
        ),
    )

    assert not result["matches"]
    assert result["numerical_mismatch_count"] > 0


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate_reordered",
        "feature_omitted",
        "feature_reordered",
        "node_rows_broadcast",
        "applicability_mask_changed",
        "missingness_mask_changed",
        "numerical_value_outside_tolerance",
    ),
)
def test_stage2_parity_rejects_contract_drift(
    parity_windows,
    mutation,
):
    example, actual, expected = parity_windows
    changed = deepcopy(actual)
    if mutation == "candidate_reordered":
        changed["node_ids"][0], changed["node_ids"][1] = (
            changed["node_ids"][1],
            changed["node_ids"][0],
        )
    elif mutation == "feature_omitted":
        changed["node_feature_names"].pop()
        for row in changed["node_features"]:
            row.pop()
    elif mutation == "feature_reordered":
        changed["node_feature_names"][0], changed[
            "node_feature_names"
        ][1] = (
            changed["node_feature_names"][1],
            changed["node_feature_names"][0],
        )
    elif mutation == "node_rows_broadcast":
        changed["node_features"] = [
            list(changed["node_features"][0])
            for _ in changed["node_features"]
        ]
    elif mutation == "applicability_mask_changed":
        index = next(
            index
            for index, name in enumerate(
                changed["node_feature_names"]
            )
            if name.endswith("__applicable")
        )
        changed["node_features"][0][index] = (
            1.0 - changed["node_features"][0][index]
        )
    elif mutation == "missingness_mask_changed":
        index = next(
            index
            for index, name in enumerate(
                changed["node_feature_names"]
            )
            if name.endswith("__missing")
        )
        changed["node_features"][0][index] = (
            1.0 - changed["node_features"][0][index]
        )
    else:
        index = next(
            index
            for index, name in enumerate(
                changed["node_feature_names"]
            )
            if not name.endswith(
                ("__applicable", "__missing")
            )
        )
        changed["node_features"][0][index] += (
            example.seal.declared_absolute_tolerance * 2.0
        )

    result = compare_v06_stage2(
        actual=changed,
        expected=expected,
        absolute_tolerance=(
            example.seal.declared_absolute_tolerance
        ),
    )
    assert not result["matches"]


def test_declared_numerical_tolerance_is_enforced(parity_windows):
    example, actual, expected = parity_windows
    changed = deepcopy(actual)
    index = next(
        index
        for index, name in enumerate(changed["node_feature_names"])
        if not name.endswith(("__applicable", "__missing"))
    )
    changed["node_features"][0][index] += (
        example.seal.declared_absolute_tolerance / 2.0
    )

    assert compare_v06_stage2(
        actual=changed,
        expected=expected,
        absolute_tolerance=example.seal.declared_absolute_tolerance,
    )["matches"]


@pytest.mark.parametrize(
    "future_boundary",
    ("event_cutoff", "knowledge_cutoff"),
)
def test_stage2_ignores_evidence_beyond_each_bitemporal_cutoff(
    sealed_example,
    future_boundary,
):
    observable = deepcopy(sealed_example.observable_scenario)
    event = deepcopy(observable["telemetry"][0])
    event["event_id"] = f"future-{future_boundary}"
    event["value"] = 999999.0
    if future_boundary == "event_cutoff":
        event["observed_at"] = (
            sealed_example.seal.event_cutoff_utc
            + timedelta(seconds=1)
        ).isoformat()
        event["ingested_at"] = (
            sealed_example.seal.knowledge_cutoff_utc
        ).isoformat()
    else:
        event["observed_at"] = (
            sealed_example.seal.event_cutoff_utc
            - timedelta(seconds=1)
        ).isoformat()
        event["ingested_at"] = (
            sealed_example.seal.knowledge_cutoff_utc
            + timedelta(seconds=1)
        ).isoformat()
    observable["telemetry"].append(event)

    actual = reconstruct_iptv_p0_v06_stage2(
        observable_scenario=observable,
        observation_start_utc=sealed_example.seal.observation_start_utc,
        event_cutoff_utc=sealed_example.seal.event_cutoff_utc,
        knowledge_cutoff_utc=(
            sealed_example.seal.knowledge_cutoff_utc
        ),
        pseudonymization_secret=SECRET,
    )
    expected = build_sealed_v06_stage2_reference(
        reference_window=sealed_example.reference_window,
        pseudonymization_secret=SECRET,
    )
    result = compare_v06_stage2(
        actual=actual,
        expected=expected,
        absolute_tolerance=(
            sealed_example.seal.declared_absolute_tolerance
        ),
    )
    assert result["matches"], result["first_differences"]
