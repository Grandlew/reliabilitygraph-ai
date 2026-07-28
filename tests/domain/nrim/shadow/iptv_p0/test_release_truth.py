from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]


def test_baseline_ledger_binds_merge_tree_and_frozen_bundle():
    ledger = json.loads(
        (ROOT / "docs/v0.8/baseline_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["baseline"]["merge_commit"] == (
        "03ec7e47ced1729e7fb24e8b0e5d44b377e8261b"
    )
    assert ledger["baseline"]["git_tree"] == (
        "4cb62a67b122bae5c11e55e3d43e81b1341c891e"
    )
    assert ledger["frozen_identities"]["bundle_sha256"] == (
        "9767bc9f3184a9e68fe99e0a31c5593a610c9dbd3b524dfa155c8bb53091ecf7"
    )
    assert ledger["real_operator_data_accessed"] is False


def test_release_ledger_lists_exactly_twenty_assignments():
    text = (ROOT / "docs/v0.8/qualification_ledger.md").read_text(
        encoding="utf-8"
    )
    assignments = [
        line for line in text.splitlines() if line[:1].isdigit() and ". " in line
    ]
    assert len(assignments) == 20
    assert assignments[0].startswith("1. ")
    assert assignments[-1].startswith("20. ")


def test_documented_claim_ceiling_is_exact():
    text = (ROOT / "docs/v0.8/qualification_ledger.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Engineering qualification infrastructure only; no real IPTV "
        "deployment has"
    ) in text
    assert "prospective incident recall" in text
    assert "production readiness" in text


def test_reference_adapter_has_no_network_or_mutation_imports():
    path = (
        ROOT
        / "app/domain/nrim/shadow/iptv_p0/batch_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imports.intersection(
        {"requests", "httpx", "socket", "urllib", "ftplib", "paramiko"}
    )


def test_canonical_evidence_has_narrow_lf_checkout_policy():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    required = {
        (
            "app/domain/nrim/examples/shadow/decision_v0_7_2/**/*.json "
            "text eol=lf"
        ),
        (
            "app/domain/nrim/examples/shadow/decision_v0_7_2/**/*.jsonl "
            "text eol=lf"
        ),
        (
            "app/domain/nrim/examples/shadow/iptv_p0_v0_8/**/*.json "
            "text eol=lf"
        ),
        (
            "app/domain/nrim/examples/shadow/iptv_p0_v0_8/**/*.jsonl "
            "text eol=lf"
        ),
    }
    assert required.issubset(set(attributes.splitlines()))


def test_no_dependency_or_lock_change_is_required():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "requests" not in pyproject
    assert "pandas" not in pyproject
    assert "numpy" not in pyproject
