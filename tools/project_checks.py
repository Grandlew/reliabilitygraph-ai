from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


NRIM_PREFIX = ("app", "domain", "nrim")


class ProjectCheckError(RuntimeError):
    """Raised when a reproducibility foundation check fails."""


@dataclass(frozen=True)
class ImportViolation:
    path: Path
    line: int
    imported: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.imported}: {self.reason}"


def _python_files(repository: Path) -> list[Path]:
    files: list[Path] = []
    for relative_root in (Path("app"), Path("tests")):
        root = repository / relative_root
        if root.exists():
            files.extend(root.rglob("*.py"))
    return sorted(files)


def _resolved_relative_import(
    repository: Path, path: Path, node: ast.ImportFrom
) -> tuple[str, ...]:
    package = path.relative_to(repository).with_suffix("").parts[:-1]
    keep = len(package) - (node.level - 1)
    if keep < 0:
        return ()
    suffix = tuple(node.module.split(".")) if node.module else ()
    return package[:keep] + suffix


def audit_imports(repository: Path) -> list[ImportViolation]:
    """Parse every source/test module and report imports escaping NRIM."""

    repository = repository.resolve()
    violations: list[ImportViolation] = []
    for path in _python_files(repository):
        relative_path = path.relative_to(repository)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            violations.append(
                ImportViolation(relative_path, 1, "<parse>", f"cannot parse: {exc}")
            )
            continue

        is_source = relative_path.parts[:3] == NRIM_PREFIX
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                level = 0
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                level = node.level
            else:
                continue

            if level and is_source:
                resolved = _resolved_relative_import(repository, path, node)
                if resolved[:3] != NRIM_PREFIX:
                    violations.append(
                        ImportViolation(
                            relative_path,
                            node.lineno,
                            "." * level + (node.module or ""),
                            "relative import escapes app.domain.nrim",
                        )
                    )
                continue

            for imported in names:
                if not imported:
                    continue
                parts = tuple(imported.split("."))
                if parts[0] == "app" and parts[:3] != NRIM_PREFIX:
                    violations.append(
                        ImportViolation(
                            relative_path,
                            node.lineno,
                            imported,
                            "absolute app import escapes app.domain.nrim",
                        )
                    )
    return violations


def require_import_boundary(repository: Path) -> None:
    violations = audit_imports(repository)
    if violations:
        details = "\n".join(str(item) for item in violations)
        raise ProjectCheckError(f"NRIM import-boundary violations:\n{details}")


def _run(
    args: Sequence[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=False
    )


def require_clean_tree(
    repository: Path,
    runner: Callable[
        [Sequence[str], Path], subprocess.CompletedProcess[str]
    ] = _run,
) -> None:
    result = runner(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], repository
    )
    if result.returncode != 0:
        raise ProjectCheckError(f"cannot inspect Git tree: {result.stderr.strip()}")
    if result.stdout.strip():
        raise ProjectCheckError(f"Git tree is not clean:\n{result.stdout.rstrip()}")


def require_lock_current(
    repository: Path,
    runner: Callable[
        [Sequence[str], Path], subprocess.CompletedProcess[str]
    ] = _run,
) -> None:
    for name in ("pyproject.toml", "uv.lock"):
        if not (repository / name).is_file():
            raise ProjectCheckError(f"required dependency file is missing: {name}")
    result = runner(["uv", "lock", "--check"], repository)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise ProjectCheckError(f"uv.lock is stale or invalid: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("imports", "clean-tree", "lock", "all"))
    parser.add_argument(
        "--repository", type=Path, default=Path.cwd(), help="repository root"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check in {"imports", "all"}:
            require_import_boundary(args.repository)
        if args.check in {"clean-tree", "all"}:
            require_clean_tree(args.repository)
        if args.check in {"lock", "all"}:
            require_lock_current(args.repository)
    except ProjectCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"project check passed: {args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
