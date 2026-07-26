from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tools.project_checks import (
    ProjectCheckError,
    require_clean_tree,
    require_lock_current,
)


SCHEMA_VERSION = 1
NODE_ID = re.compile(r"^(?:tests[/\\]).+::.+$")


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be generated or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_value(repository: Path, argument: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", argument],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EvidenceError(result.stderr.strip() or f"cannot resolve {argument}")
    return result.stdout.strip()


def environment_snapshot() -> dict[str, Any]:
    packages = sorted(
        (
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
        key=lambda item: (item["name"].lower(), item["version"]),
    )
    return {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def read_test_ids(path: Path) -> list[str]:
    ids = sorted(
        {
            line.strip().replace("\\", "/")
            for line in path.read_text(encoding="utf-8").splitlines()
            if NODE_ID.match(line.strip())
        }
    )
    if not ids:
        raise EvidenceError(f"no pytest node IDs found in {path}")
    return ids


def junit_test_count(path: Path) -> int:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise EvidenceError(f"invalid JUnit XML {path}: {exc}") from exc
    cases = root.findall(".//testcase")
    if not cases:
        raise EvidenceError(f"JUnit XML has no test cases: {path}")
    return len(cases)


def generate_manifest(
    repository: Path, junit: Path, test_ids_path: Path
) -> dict[str, Any]:
    repository = repository.resolve()
    require_clean_tree(repository)
    require_lock_current(repository)
    junit = junit.resolve()
    test_ids_path = test_ids_path.resolve()
    test_ids = read_test_ids(test_ids_path)
    junit_count = junit_test_count(junit)
    if junit_count != len(test_ids):
        raise EvidenceError(
            f"test count mismatch: JUnit={junit_count}, node IDs={len(test_ids)}"
        )
    environment = environment_snapshot()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "commit": git_value(repository, "HEAD"),
            "tree": git_value(repository, "HEAD^{tree}"),
        },
        "lock": {
            "path": "uv.lock",
            "sha256": sha256_file(repository / "uv.lock"),
        },
        "environment": environment,
        "environment_sha256": sha256_json(environment),
        "tests": {
            "count": len(test_ids),
            "ids": test_ids,
            "ids_sha256": sha256_json(test_ids),
        },
        "junit": {
            "path": junit.relative_to(repository).as_posix(),
            "sha256": sha256_file(junit),
        },
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_manifest(repository: Path, manifest_path: Path) -> None:
    repository = repository.resolve()
    require_clean_tree(repository)
    require_lock_current(repository)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvidenceError(f"invalid evidence manifest: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("unsupported evidence schema version")

    expected = {
        "commit": git_value(repository, "HEAD"),
        "tree": git_value(repository, "HEAD^{tree}"),
    }
    if manifest.get("repository") != expected:
        raise EvidenceError("repository commit/tree binding does not match")
    if manifest.get("lock", {}).get("sha256") != sha256_file(
        repository / "uv.lock"
    ):
        raise EvidenceError("uv.lock hash does not match")

    environment = environment_snapshot()
    if manifest.get("environment_sha256") != sha256_json(environment):
        raise EvidenceError("execution environment hash does not match")
    if manifest.get("environment") != environment:
        raise EvidenceError("execution environment details do not match")

    test_ids = manifest.get("tests", {}).get("ids")
    if not isinstance(test_ids, list) or not test_ids:
        raise EvidenceError("test ID binding is missing")
    if manifest["tests"].get("count") != len(test_ids):
        raise EvidenceError("test ID count does not match")
    if manifest["tests"].get("ids_sha256") != sha256_json(test_ids):
        raise EvidenceError("test ID hash does not match")

    junit_relative = manifest.get("junit", {}).get("path")
    if not isinstance(junit_relative, str):
        raise EvidenceError("JUnit path binding is missing")
    junit = repository / junit_relative
    if manifest["junit"].get("sha256") != sha256_file(junit):
        raise EvidenceError("JUnit hash does not match")
    if junit_test_count(junit) != len(test_ids):
        raise EvidenceError("JUnit test count does not match bound test IDs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "verify"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--test-ids", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "generate":
            if not all((args.junit, args.test_ids, args.output)):
                raise EvidenceError(
                    "generate requires --junit, --test-ids, and --output"
                )
            manifest = generate_manifest(
                args.repository, args.junit, args.test_ids
            )
            write_manifest(manifest, args.output)
            print(f"wrote evidence manifest: {args.output}")
        else:
            if args.manifest is None:
                raise EvidenceError("verify requires --manifest")
            verify_manifest(args.repository, args.manifest)
            print(f"evidence verified: {args.manifest}")
    except (EvidenceError, ProjectCheckError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

