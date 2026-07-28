from __future__ import annotations

import json
from pathlib import Path

from app.domain.nrim.shadow.hashing import file_hash
from app.domain.nrim.shadow.iptv_p0.qualification import (
    verify_evidence_bundle,
)
from app.domain.nrim.shadow.iptv_p0.repository_hygiene import (
    scan_fixture_tree,
)
from app.domain.nrim.shadow.iptv_p0.synthetic_fixture import (
    build_synthetic_trusted_signer_registry,
    generate_fixture,
)


ROOT = Path(__file__).resolve().parents[5]
FIXTURE = (
    ROOT
    / "app/domain/nrim/examples/shadow/iptv_p0_v0_8"
)


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_checked_in_fixture_manifest_verifies_every_file():
    manifest = json.loads(
        (FIXTURE / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["terminal_decision"] == "REAL_DATA_REQUIRED"
    for relative, expected in manifest["files"].items():
        assert file_hash(FIXTURE / relative) == expected


def test_checked_in_qualification_bundle_verifies():
    manifest = verify_evidence_bundle(
        FIXTURE / "expected",
        trusted_signer_registry=(
            build_synthetic_trusted_signer_registry()
        ),
    )
    assert manifest["decision"] == "REAL_DATA_REQUIRED"
    assert manifest["synthetic_evidence_only"] is True


def test_checked_in_fixture_regenerates_byte_for_byte(tmp_path):
    generated = tmp_path / "iptv_p0_v0_8"
    generate_fixture(generated)
    assert _bytes(generated) == _bytes(FIXTURE)


def test_checked_in_fixture_contains_no_identifier_or_secret_findings():
    assert scan_fixture_tree(FIXTURE) == ()


def test_registered_mutation_inventory_is_complete():
    value = json.loads(
        (FIXTURE / "synthetic/mutations.json").read_text(encoding="utf-8")
    )
    ids = {
        item["id"] for item in value["registered_mutations"]
    }
    assert ids == {
        "unit_swap",
        "timezone_removed",
        "negative_sign",
        "aggregation_profile_change",
        "required_signal_missing",
        "applicability_unknown",
        "topology_reversed",
    }


def test_all_checked_in_json_and_jsonl_are_lf_only():
    for path in FIXTURE.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".md"}:
            payload = path.read_bytes()
            assert b"\r\n" not in payload
            assert payload.endswith(b"\n")
