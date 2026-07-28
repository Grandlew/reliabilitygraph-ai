from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.contracts import (
    create_ed25519_signature,
    verify_signature,
)
from app.domain.nrim.shadow.iptv_p0.feature_reconstruction import (
    reconstruct_frozen_features,
)

BASE_TIME = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)


def test_stage2_is_component_local_not_broadcast(
    records,
    registry,
    topology_snapshot,
):
    result = reconstruct_frozen_features(
        records=records,
        registry=registry,
        topology=topology_snapshot,
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )
    width = len(result.stage2.feature_names)
    rows = tuple(
        result.stage2.values[index : index + width]
        for index in range(0, len(result.stage2.values), width)
    )

    assert len(set(rows)) > 1
    assert len(result.per_node_lineage) == len(rows) * width


def test_qualification_request_does_not_accept_authoritative_metric_scalars(
    qualification_request,
):
    assert "metrics" not in type(qualification_request).model_fields
    assert not hasattr(qualification_request, "metrics")
    schema = type(qualification_request).model_json_schema()
    serialized_schema = str(schema)
    for forbidden in (
        "collector_mapping_fraction",
        "semantic_mutation_rejection_fraction",
        "topology_reconstruction_fraction",
        "topology_mutation_rejection_fraction",
        "required_feature_fraction",
        "incident_alignment_fraction",
        "root_cause_mapping_fraction",
        "future_leakage_count",
        "identifier_leakage_count",
        "evidence_determinism_fraction",
        "tamper_detection_passed",
    ):
        assert forbidden not in serialized_schema


def test_measured_results_are_immutable(qualification_request):
    with pytest.raises(ValidationError):
        qualification_request.measurements.semantic.result_id = "rewritten"


def test_embedded_public_key_is_not_a_trust_decision():
    payload_sha256 = "a" * 64
    signature = create_ed25519_signature(
        payload_sha256=payload_sha256,
        private_key_bytes=bytes(range(1, 33)),
        key_id="self-asserted-key",
    )

    assert not verify_signature(
        signature,
        expected_payload_sha256=payload_sha256,
    )
