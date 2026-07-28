from __future__ import annotations

import pytest

from app.domain.nrim.shadow.iptv_p0.repository_hygiene import (
    scan_fixture_tree,
)


def _tree(tmp_path, content):
    (tmp_path / "SYNTHETIC_ONLY.md").write_text(
        "Synthetic only; not production evidence.\n",
        encoding="utf-8",
    )
    (tmp_path / "records.jsonl").write_text(content, encoding="utf-8")
    return tmp_path


def test_safe_synthetic_fixture_has_no_findings(tmp_path):
    root = _tree(
        tmp_path,
        '{"source_id":"src_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n',
    )
    assert scan_fixture_tree(root) == ()


def test_fixture_tree_requires_synthetic_marker(tmp_path):
    with pytest.raises(ValueError, match="marker"):
        scan_fixture_tree(tmp_path)


@pytest.mark.parametrize(
    "value,rule",
    (
        ("operator@example.com", "email"),
        ("192.168.10.22", "ipv4"),
        ("00:11:22:33:44:55", "mac"),
        ("api_key=unsafe-value", "credential"),
        ("-----BEGIN PRIVATE KEY-----", "private_key"),
    ),
)
def test_repository_hygiene_detects_identifier_and_secret_patterns(
    tmp_path,
    value,
    rule,
):
    findings = scan_fixture_tree(_tree(tmp_path, value + "\n"))
    assert any(item.rule == rule for item in findings)
    assert all(value not in item.model_dump_json() for item in findings)


def test_hygiene_finding_is_deterministic(tmp_path):
    root = _tree(tmp_path, "operator@example.com\n")
    assert scan_fixture_tree(root) == scan_fixture_tree(root)
