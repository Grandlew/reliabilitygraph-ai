from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from tools.project_checks import (
    ProjectCheckError,
    audit_imports,
    require_clean_tree,
    require_import_boundary,
    require_lock_current,
)


def completed(
    args: Sequence[str],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_repository_import_boundary_is_clean() -> None:
    require_import_boundary(Path.cwd())


def test_import_boundary_rejects_absolute_app_escape(tmp_path: Path) -> None:
    source = tmp_path / "app/domain/nrim/bad.py"
    source.parent.mkdir(parents=True)
    source.write_text("from app.services import presales\n", encoding="utf-8")

    violations = audit_imports(tmp_path)

    assert len(violations) == 1
    assert violations[0].imported == "app.services"


def test_import_boundary_rejects_relative_escape(tmp_path: Path) -> None:
    source = tmp_path / "app/domain/nrim/bad.py"
    source.parent.mkdir(parents=True)
    source.write_text("from ..presales import quote\n", encoding="utf-8")

    violations = audit_imports(tmp_path)

    assert len(violations) == 1
    assert "relative import escapes" in violations[0].reason


def test_clean_tree_rejects_modified_or_untracked_files(tmp_path: Path) -> None:
    def dirty_runner(
        args: Sequence[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        return completed(args, stdout=" M app/domain/nrim/models.py\n?? output.xml\n")

    with pytest.raises(ProjectCheckError, match="not clean"):
        require_clean_tree(tmp_path, runner=dirty_runner)


def test_lock_check_rejects_stale_lock(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    def stale_runner(
        args: Sequence[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        return completed(args, returncode=2, stderr="lock needs to be updated")

    with pytest.raises(ProjectCheckError, match="stale or invalid"):
        require_lock_current(tmp_path, runner=stale_runner)


def test_lock_check_rejects_missing_lock(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    with pytest.raises(ProjectCheckError, match="uv.lock"):
        require_lock_current(tmp_path)

