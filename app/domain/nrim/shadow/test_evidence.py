from __future__ import annotations

import argparse
import json
import platform
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from .hashing import bytes_hash, file_hash


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def build_test_evidence(
    *,
    root: Path,
    junit_path: Path,
) -> dict[str, Any]:
    suite = ET.parse(junit_path).getroot()
    cases = list(suite.iter("testcase"))
    node_ids = [
        f"{item.get('classname', '')}::{item.get('name', '')}"
        for item in cases
    ]
    shadow = [
        item for item in node_ids if ".nrim.shadow." in item
    ]
    nrim = [item for item in node_ids if ".nrim." in item]
    failures = list(suite.iter("failure"))
    errors = list(suite.iter("error"))
    skipped = list(suite.iter("skipped"))
    status = _git(root, "status", "--porcelain=v1").decode("utf-8")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_changed_names = _git(
        root,
        "diff",
        "--name-only",
        "HEAD",
        "--",
    ).decode("utf-8").splitlines()
    untracked_names = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
    ).decode("utf-8").splitlines()
    evidence_directory = (
        root
        / "app"
        / "domain"
        / "nrim"
        / "examples"
        / "shadow"
        / "oelr_v0_7_1"
        / "test-results"
    ).resolve()

    def is_source_artifact(name: str) -> bool:
        path = (root / name).resolve()
        return (
            path != junit_path.resolve()
            and evidence_directory not in path.parents
        )

    changed_source_names = sorted(
        {
            name
            for name in (*tracked_changed_names, *untracked_names)
            if is_source_artifact(name)
        }
    )
    source_manifest = "\n".join(
        (
            f"{name}:{file_hash(root / name)}"
            if (root / name).is_file()
            else f"{name}:DELETED"
        )
        for name in changed_source_names
    ).encode("utf-8")
    source_state_hash = bytes_hash(
        f"HEAD:{head}".encode("ascii")
        + b"\0CHANGED_SOURCE_FILES\0"
        + source_manifest
    )
    return {
        "schema_version": "0.7.1-oelr",
        "suite": "full_repository",
        "test_count": len(cases),
        "failure_count": len(failures),
        "error_count": len(errors),
        "skipped_count": len(skipped),
        "passed_count": (
            len(cases) - len(failures) - len(errors) - len(skipped)
        ),
        "scope_counts": {
            "shadow_subset": len(shadow),
            "nrim_subset_including_shadow": len(nrim),
            "full_suite_including_nrim": len(cases),
        },
        "scope_relationship": (
            "shadow_subset is contained in nrim_subset; nrim_subset is "
            "contained in full_suite. Counts must not be summed."
        ),
        "node_ids": node_ids,
        "junit_sha256": file_hash(junit_path),
        "git_commit": head,
        "working_tree_clean": not bool(status.strip()),
        "working_tree_source_clean": not changed_source_names,
        "working_tree_source_state_sha256": source_state_hash,
        "python_version": platform.python_version(),
        "pytest_version": pytest.__version__,
        "requirements_sha256": file_hash(root / "requirements.txt"),
        "truth_note": (
            "JUnit evidence proves the recorded software test run only. It "
            "does not establish real telemetry or operational efficacy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    evidence = build_test_evidence(
        root=root,
        junit_path=(root / arguments.junit).resolve(),
    )
    output = (root / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
