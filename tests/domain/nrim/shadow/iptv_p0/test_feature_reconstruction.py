from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.simulation.feature_schema import SIGNAL_NAMES
from app.domain.nrim.shadow.iptv_p0.feature_reconstruction import (
    FeatureEvidenceState,
    ReconstructionState,
    TensorEvidence,
    reconstruct_frozen_features,
)
from app.domain.nrim.shadow.iptv_p0.signal_registry import (
    AvailabilityClass,
    SignalRegistry,
    SignalRequirement,
    SupportConsequence,
)


BASE_TIME = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)


def _run(records, registry, topology_snapshot, predicates=None):
    return reconstruct_frozen_features(
        records=tuple(records),
        registry=registry,
        topology=topology_snapshot,
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
        conditional_predicates=predicates,
    )


def _replace_requirement(registry, signal_name, **updates):
    changed = []
    for item in registry.requirements:
        if item.signal_name == signal_name:
            value = item.model_dump(mode="json")
            value.update(updates)
            item = SignalRequirement.model_validate(value)
        changed.append(item)
    return SignalRegistry(
        registry_id=registry.registry_id,
        registry_version=registry.registry_version,
        requirements=tuple(changed),
    )


def test_complete_feature_dry_run_builds_all_three_tensors(
    records,
    registry,
    topology_snapshot,
):
    result = _run(records, registry, topology_snapshot)
    assert result.state is ReconstructionState.COMPLETE
    assert result.stage1.shape == (len(SIGNAL_NAMES),)
    assert result.residual.shape == (len(SIGNAL_NAMES),)
    assert result.stage2.shape == (
        len(topology_snapshot.nodes),
        len(SIGNAL_NAMES),
    )
    assert len(result.lineage) == len(SIGNAL_NAMES)


def test_feature_dry_run_never_invokes_inference_or_tuning(
    records,
    registry,
    topology_snapshot,
):
    result = _run(records, registry, topology_snapshot)
    assert result.inference_call_count == 0
    assert result.tuning_event_count == 0


def test_feature_dry_run_is_deterministic(
    records,
    registry,
    topology_snapshot,
):
    first = _run(records, registry, topology_snapshot)
    second = _run(tuple(reversed(records)), registry, topology_snapshot)
    assert first.content_hash == second.content_hash


@pytest.mark.parametrize("missing", SIGNAL_NAMES)
def test_each_missing_required_frozen_signal_blocks(
    records,
    registry,
    topology_snapshot,
    missing,
):
    result = _run(
        [item for item in records if item.metric_name != missing],
        registry,
        topology_snapshot,
    )
    assert result.state is ReconstructionState.BLOCKED
    assert f"BLOCKED:{missing}:MISSING" in result.reason_codes
    assert result.required_feature_fraction < 1.0


def test_future_event_time_is_not_visible(
    records,
    registry,
    topology_snapshot,
):
    first = records[0].model_copy(
        update={
            "event_time_utc": BASE_TIME + timedelta(seconds=1),
            "observation_time_utc": BASE_TIME + timedelta(seconds=2),
            "ingestion_time_utc": BASE_TIME + timedelta(seconds=3),
        }
    )
    result = _run((first, *records[1:]), registry, topology_snapshot)
    assert result.state is ReconstructionState.BLOCKED


def test_future_knowledge_is_not_visible(
    records,
    registry,
    topology_snapshot,
):
    first = records[0].model_copy(
        update={"ingestion_time_utc": BASE_TIME + timedelta(seconds=1)}
    )
    result = _run((first, *records[1:]), registry, topology_snapshot)
    assert result.state is ReconstructionState.BLOCKED


def test_conditional_false_is_explicitly_not_applicable(
    records,
    registry,
    topology_snapshot,
):
    signal = SIGNAL_NAMES[0]
    conditional = _replace_requirement(
        registry,
        signal,
        availability=AvailabilityClass.CONDITIONAL,
        applicability_predicate="storage_is_present",
    )
    result = _run(
        [item for item in records if item.metric_name != signal],
        conditional,
        topology_snapshot,
        {signal: False},
    )
    lineage = next(
        item for item in result.lineage if item.feature_name == signal
    )
    assert result.state is ReconstructionState.COMPLETE
    assert lineage.state is FeatureEvidenceState.NOT_APPLICABLE


def test_conditional_unknown_predicate_blocks(
    records,
    registry,
    topology_snapshot,
):
    signal = SIGNAL_NAMES[0]
    conditional = _replace_requirement(
        registry,
        signal,
        availability=AvailabilityClass.CONDITIONAL,
        applicability_predicate="storage_is_present",
    )
    result = _run(
        [item for item in records if item.metric_name != signal],
        conditional,
        topology_snapshot,
        {signal: None},
    )
    assert result.state is ReconstructionState.BLOCKED


def test_optional_absence_does_not_impute_health(
    records,
    registry,
    topology_snapshot,
):
    signal = SIGNAL_NAMES[0]
    optional = _replace_requirement(
        registry,
        signal,
        availability=AvailabilityClass.OPTIONAL,
        support_consequence=SupportConsequence.UNKNOWN,
        source_ids=(),
    )
    result = _run(
        [item for item in records if item.metric_name != signal],
        optional,
        topology_snapshot,
    )
    index = SIGNAL_NAMES.index(signal)
    assert result.state is ReconstructionState.COMPLETE
    assert result.stage1.values[index] is None
    assert result.stage1.observed_mask[index] == 0


def test_unavailable_signal_routes_to_escalate(
    records,
    registry,
    topology_snapshot,
):
    signal = SIGNAL_NAMES[0]
    unavailable = _replace_requirement(
        registry,
        signal,
        availability=AvailabilityClass.UNAVAILABLE,
        support_consequence=SupportConsequence.ESCALATE,
        source_ids=(),
    )
    result = _run(records, unavailable, topology_snapshot)
    assert result.state is ReconstructionState.ESCALATE
    assert f"ESCALATE:{signal}:UNAVAILABLE" in result.reason_codes


def test_residual_tensor_uses_only_visible_prior_values(
    records,
    registry,
    topology_snapshot,
    record_factory,
):
    signal = SIGNAL_NAMES[0]
    current = next(item for item in records if item.metric_name == signal)
    prior = record_factory(
        metric_name=signal,
        sequence="prior",
        value=float(current.metric_value) - 2.0,
        event_time=current.event_time_utc - timedelta(minutes=1),
    )
    result = _run((prior, *records), registry, topology_snapshot)
    index = SIGNAL_NAMES.index(signal)
    assert result.residual.values[index] == 2.0


@pytest.mark.parametrize(
    "values,mask",
    (
        ((None,), (1,)),
        ((1.0,), (0,)),
    ),
)
def test_tensor_evidence_rejects_mask_value_conflict(values, mask):
    with pytest.raises(ValidationError):
        TensorEvidence(
            tensor_name="invalid",
            feature_names=("feature",),
            shape=(1,),
            values=values,
            observed_mask=mask,
        )


def test_feature_lineage_contains_only_record_hashes_not_raw_payload(
    records,
    registry,
    topology_snapshot,
):
    result = _run(records, registry, topology_snapshot)
    serialized = result.model_dump_json()
    assert "source_record_sha256" in serialized
    assert "metric_value" not in serialized
    assert "source_sequence_id" not in serialized
    assert all(
        item.topology_snapshot_sha256 == topology_snapshot.content_hash
        for item in result.per_node_lineage
    )


def test_stage2_reconstructs_component_local_applicability_and_time(
    records,
    record_factory,
    registry,
    topology_snapshot,
):
    local_component = "cmp_" + "c" * 16
    nodes = list(topology_snapshot.nodes)
    nodes[0] = nodes[0].model_copy(
        update={
            "observation_component_pseudonym": local_component,
            "applicable_signals": tuple(SIGNAL_NAMES[:2]),
        }
    )
    topology = topology_snapshot.model_copy(update={"nodes": tuple(nodes)})
    local = record_factory(
        metric_name=SIGNAL_NAMES[0],
        sequence="local-source-headend",
        value=999.0,
        component=local_component,
    )
    future_visible = record_factory(
        metric_name=SIGNAL_NAMES[1],
        sequence="future-local-source-headend",
        value=888.0,
        component=local_component,
        ingestion_time=BASE_TIME + timedelta(seconds=1),
    )

    result = _run(
        (*records, local, future_visible),
        registry,
        topology,
    )
    width = len(SIGNAL_NAMES)
    source_row = result.stage2.values[:width]
    aggregation_row = result.stage2.values[2 * width : 3 * width]

    assert source_row[0] == 999.0
    assert aggregation_row[0] == records[0].metric_value
    assert source_row[1] is None
    assert source_row[2] is None
    source_lineage = result.per_node_lineage[:width]
    assert source_lineage[0].state is FeatureEvidenceState.OBSERVED
    assert source_lineage[1].state is FeatureEvidenceState.MISSING
    assert source_lineage[1].source_record_sha256 == ()
    assert source_lineage[2].state is FeatureEvidenceState.NOT_APPLICABLE
    assert all(
        item.topology_snapshot_sha256 == topology.content_hash
        and item.topology_capture_ids == topology.applied_capture_ids
        for item in source_lineage
    )
