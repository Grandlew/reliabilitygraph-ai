from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.decision_events import (
    DecisionEvent,
    SupportAssessedPayload,
)
from app.domain.nrim.shadow.privacy import PrivacyViolation


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_scope_ledger_binds_frozen_identities() -> None:
    root = _root()
    scope = json.loads(
        (
            root
            / "app/domain/nrim/examples/shadow/decision_v0_7_2/scope.json"
        ).read_text(encoding="utf-8")
    )
    commitment = json.loads(
        (
            root
            / "app/domain/nrim/examples/shadow/v0_7_0/"
            "frozen_bundle/bundle_commitment.json"
        ).read_text(encoding="utf-8")
    )
    for field in (
        "bundle_manifest_sha256",
        "bundle_sha256",
        "feature_schema_sha256",
        "policy_sha256",
    ):
        assert scope["frozen_identities"][field] == commitment[field]


@pytest.mark.parametrize(
    "capability",
    [
        "active_query",
        "automated_remediation",
        "configuration_write",
        "customer_notification",
        "restart",
        "ticket_write",
    ],
)
def test_scope_ledger_forbids_operational_authority(
    capability: str,
) -> None:
    scope = json.loads(
        (
            _root()
            / "app/domain/nrim/examples/shadow/decision_v0_7_2/scope.json"
        ).read_text(encoding="utf-8")
    )
    assert capability in scope["forbidden_capabilities"]


def test_scope_ledger_contains_synthetic_disclaimer() -> None:
    scope = json.loads(
        (
            _root()
            / "app/domain/nrim/examples/shadow/decision_v0_7_2/scope.json"
        ).read_text(encoding="utf-8")
    )
    ledger = (
        _root() / "docs/v0.7.2/claim_ledger.md"
    ).read_text(encoding="utf-8")
    assert scope["truth_status"] == "synthetic_software_evidence_only"
    assert "does not establish real IPTV accuracy" in ledger


def test_event_payload_rejects_direct_identifier_content(
    decision_events,
) -> None:
    event = decision_events[1]
    payload = event.payload.model_copy(
        update={"reason_codes": ("operator@example.com",)}
    )
    with pytest.raises((PrivacyViolation, ValidationError)):
        DecisionEvent.create(
            event_type=event.event_type,
            deployment_pseudonym=event.deployment_pseudonym,
            decision_id=event.decision_id,
            decision_cutoff_utc=event.decision_cutoff_utc,
            event_time_utc=event.event_time_utc,
            recorded_at_utc=event.recorded_at_utc,
            stream_sequence=event.stream_sequence,
            correlation_id=event.correlation_id,
            payload=payload,
            previous_event_sha256=event.previous_event_sha256,
        )


@pytest.mark.parametrize(
    "module_name",
    [
        "candidate_evidence.py",
        "decision_events.py",
        "decision_envelope.py",
        "decision_projection.py",
    ],
)
def test_pure_decision_modules_do_not_import_store_or_api(
    module_name: str,
) -> None:
    path = _root() / "app/domain/nrim/shadow" / module_name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        name.endswith("decision_store")
        or name.endswith("decision_api")
        or name.endswith(".store")
        for name in imports
    )


def test_read_api_source_contains_no_operational_connectors() -> None:
    source = (
        _root() / "app/domain/nrim/shadow/decision_api.py"
    ).read_text(encoding="utf-8").lower()
    for forbidden in (
        "restart_service",
        "create_ticket",
        "write_config",
        "suppress_alarm",
        "notify_customer",
    ):
        assert forbidden not in source
