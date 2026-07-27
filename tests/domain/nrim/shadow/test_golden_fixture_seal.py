from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.nrim.shadow.golden_replay import (
    run_golden_replay,
    verify_golden_fixture,
)


@pytest.fixture
def fixture_root() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "app/domain/nrim/examples/shadow/decision_v0_7_2"
    )


def test_checked_in_golden_fixture_verifies(fixture_root: Path) -> None:
    manifest = verify_golden_fixture(fixture_root)
    assert manifest["portable_paths_only"] is True
    assert manifest["truth_status"] == "synthetic_software_evidence_only"


@pytest.mark.parametrize(
    "relative",
    [
        "input/events.jsonl",
        "expected/events.jsonl",
        "expected/snapshots.jsonl",
        "expected/manifest.json",
        "scope.json",
    ],
)
def test_fixture_detects_any_committed_file_mutation(
    tmp_path,
    fixture_root: Path,
    relative: str,
) -> None:
    import shutil

    copied = tmp_path / "fixture"
    shutil.copytree(fixture_root, copied)
    path = copied / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="commitment differs"):
        verify_golden_fixture(copied)


def test_fixture_replay_matches_checked_in_expected_bytes(
    tmp_path,
    fixture_root: Path,
) -> None:
    result = run_golden_replay(
        input_path=fixture_root / "input/events.jsonl",
        output_directory=tmp_path / "output",
    )
    assert result.events_sha256 == json.loads(
        (fixture_root / "expected/manifest.json").read_text(encoding="utf-8")
    )["events_sha256"]
    for name in ("events.jsonl", "snapshots.jsonl", "manifest.json"):
        assert (tmp_path / "output" / name).read_bytes() == (
            fixture_root / "expected" / name
        ).read_bytes()


def test_fixture_contains_no_live_absolute_provenance_path(
    fixture_root: Path,
) -> None:
    input_text = (fixture_root / "input/events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "C:\\" not in input_text
    assert "/home/" not in input_text
    assert "/Users/" not in input_text


def test_fixture_disclaimer_is_conspicuous(fixture_root: Path) -> None:
    disclaimer = (fixture_root / "SYNTHETIC_ONLY.md").read_text(
        encoding="utf-8"
    ).lower()
    assert "synthetic fixture only" in disclaimer
    assert "not production evidence" in disclaimer
    assert "authorizes no operational action" in disclaimer
