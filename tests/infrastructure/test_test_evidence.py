from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.test_evidence as evidence


def write_junit(path: Path, names: tuple[str, ...] = ("one", "two")) -> None:
    cases = "".join(
        f'<testcase classname="tests.test_sample" name="{name}" />'
        for name in names
    )
    path.write_text(
        f'<testsuites tests="{len(names)}"><testsuite>{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )


@pytest.fixture
def evidence_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    junit = tmp_path / "artifacts/test-evidence/junit.xml"
    junit.parent.mkdir(parents=True)
    write_junit(junit)
    test_ids = junit.parent / "test-ids.txt"
    test_ids.write_text(
        "tests/test_sample.py::test_one\n"
        "tests/test_sample.py::test_two\n"
        "\n2 tests collected\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evidence, "require_clean_tree", lambda repository: None)
    monkeypatch.setattr(evidence, "require_lock_current", lambda repository: None)
    monkeypatch.setattr(
        evidence,
        "git_value",
        lambda repository, argument: {
            "HEAD": "commit-id",
            "HEAD^{tree}": "tree-id",
        }[argument],
    )
    monkeypatch.setattr(
        evidence,
        "environment_snapshot",
        lambda: {
            "implementation": "CPython",
            "python": "3.12.0",
            "platform": "test-platform",
            "packages": [{"name": "pytest", "version": "9.0.0"}],
        },
    )
    return tmp_path, junit


def manifest_path_for(repository: Path, junit: Path) -> Path:
    test_ids = junit.parent / "test-ids.txt"
    manifest = evidence.generate_manifest(repository, junit, test_ids)
    path = junit.parent / "manifest.json"
    evidence.write_manifest(manifest, path)
    return path


def test_generated_evidence_verifies(
    evidence_repository: tuple[Path, Path],
) -> None:
    repository, junit = evidence_repository
    manifest = manifest_path_for(repository, junit)

    evidence.verify_manifest(repository, manifest)


def test_evidence_rejects_changed_junit(
    evidence_repository: tuple[Path, Path],
) -> None:
    repository, junit = evidence_repository
    manifest = manifest_path_for(repository, junit)
    write_junit(junit, ("one", "tampered"))

    with pytest.raises(evidence.EvidenceError, match="JUnit hash"):
        evidence.verify_manifest(repository, manifest)


def test_evidence_rejects_changed_lock(
    evidence_repository: tuple[Path, Path],
) -> None:
    repository, junit = evidence_repository
    manifest = manifest_path_for(repository, junit)
    (repository / "uv.lock").write_text("version = 2\n", encoding="utf-8")

    with pytest.raises(evidence.EvidenceError, match="uv.lock hash"):
        evidence.verify_manifest(repository, manifest)


def test_evidence_rejects_changed_test_ids(
    evidence_repository: tuple[Path, Path],
) -> None:
    repository, junit = evidence_repository
    manifest_path = manifest_path_for(repository, junit)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tests"]["ids"][0] = "tests/test_sample.py::test_tampered"
    evidence.write_manifest(manifest, manifest_path)

    with pytest.raises(evidence.EvidenceError, match="test ID hash"):
        evidence.verify_manifest(repository, manifest_path)


def test_evidence_rejects_changed_environment(
    evidence_repository: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, junit = evidence_repository
    manifest = manifest_path_for(repository, junit)
    monkeypatch.setattr(
        evidence,
        "environment_snapshot",
        lambda: {
            "implementation": "CPython",
            "python": "3.12.1",
            "platform": "test-platform",
            "packages": [],
        },
    )

    with pytest.raises(evidence.EvidenceError, match="environment hash"):
        evidence.verify_manifest(repository, manifest)

