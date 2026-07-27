from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from app.domain.nrim.shadow.decision_events import (
    AmendmentReason,
    DecisionEvent,
    DecisionEventType,
    EvidenceAmendedPayload,
)
from app.domain.nrim.shadow.golden_replay import (
    export_events,
    load_event_jsonl,
    run_golden_replay,
)
from app.domain.nrim.shadow.hashing import canonical_hash


def _input(tmp_path: Path, events, name: str = "input.jsonl") -> Path:
    path = tmp_path / name
    export_events(tuple(events), output_path=path)
    return path


def test_two_clean_replays_are_byte_identical(
    tmp_path,
    decision_events,
) -> None:
    source = _input(tmp_path, decision_events)
    first = run_golden_replay(
        input_path=source,
        output_directory=tmp_path / "first",
    )
    second = run_golden_replay(
        input_path=source,
        output_directory=tmp_path / "second",
    )
    assert first.events_sha256 == second.events_sha256
    assert first.snapshots_sha256 == second.snapshots_sha256
    assert (tmp_path / "first/events.jsonl").read_bytes() == (
        tmp_path / "second/events.jsonl"
    ).read_bytes()


def test_replay_is_independent_of_input_file_order(
    tmp_path,
    decision_events,
) -> None:
    normal = _input(tmp_path, decision_events, "normal.jsonl")
    reversed_path = _input(
        tmp_path,
        reversed(decision_events),
        "reversed.jsonl",
    )
    first = run_golden_replay(
        input_path=normal,
        output_directory=tmp_path / "normal",
    )
    second = run_golden_replay(
        input_path=reversed_path,
        output_directory=tmp_path / "reversed",
    )
    assert first.events_sha256 == second.events_sha256
    assert first.snapshots_sha256 == second.snapshots_sha256


def test_replay_restart_overwrites_with_identical_bytes(
    tmp_path,
    decision_events,
) -> None:
    source = _input(tmp_path, decision_events)
    output = tmp_path / "output"
    run_golden_replay(input_path=source, output_directory=output)
    before = {
        path.name: path.read_bytes() for path in output.iterdir()
    }
    run_golden_replay(input_path=source, output_directory=output)
    assert before == {
        path.name: path.read_bytes() for path in output.iterdir()
    }


def test_replay_remains_portable_after_fixture_relocation(
    tmp_path,
    decision_events,
) -> None:
    source = _input(tmp_path, decision_events)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    moved = relocated / source.name
    source.rename(moved)
    result = run_golden_replay(
        input_path=moved,
        output_directory=relocated / "output",
    )
    assert result.event_count == len(decision_events)


def test_replay_rejects_crlf_jsonl(tmp_path, decision_events) -> None:
    source = _input(tmp_path, decision_events)
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="CRLF"):
        load_event_jsonl(source)


def test_replay_rejects_nonterminated_record(
    tmp_path,
    decision_events,
) -> None:
    source = _input(tmp_path, decision_events)
    source.write_bytes(source.read_bytes().removesuffix(b"\n"))
    with pytest.raises(ValueError, match="end with LF"):
        load_event_jsonl(source)


def test_replay_outputs_canonical_lf_jsonl(
    tmp_path,
    decision_events,
) -> None:
    source = _input(tmp_path, decision_events)
    run_golden_replay(
        input_path=source,
        output_directory=tmp_path / "output",
    )
    for name in ("events.jsonl", "snapshots.jsonl"):
        payload = (tmp_path / "output" / name).read_bytes()
        assert payload.endswith(b"\n")
        assert b"\r\n" not in payload


def test_manifest_contains_no_tuning_or_production_claim(
    tmp_path,
    decision_events,
) -> None:
    source = _input(tmp_path, decision_events)
    run_golden_replay(
        input_path=source,
        output_directory=tmp_path / "output",
    )
    manifest = json.loads(
        (tmp_path / "output/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["truth_status"] == "synthetic_software_evidence_only"
    assert "tuning" not in manifest


def test_late_facts_are_retrospective_events_only(
    tmp_path,
    decision_events,
) -> None:
    target = decision_events[3]
    amendment = DecisionEvent.create(
        event_type=DecisionEventType.EVIDENCE_AMENDED,
        deployment_pseudonym=target.deployment_pseudonym,
        decision_id=target.decision_id,
        decision_cutoff_utc=target.decision_cutoff_utc,
        event_time_utc=target.decision_cutoff_utc + timedelta(hours=1),
        recorded_at_utc=target.recorded_at_utc + timedelta(hours=2),
        stream_sequence=9,
        causation_event_id=decision_events[-1].event_id,
        correlation_id=target.correlation_id,
        payload=EvidenceAmendedPayload(
            target_event_id=target.event_id,
            reason_code=AmendmentReason.LATE_OBSERVATION,
            evidence_references=("late-observation",),
            amendment_sha256=canonical_hash({"late": True}),
        ),
        previous_event_sha256=decision_events[-1].event_sha256,
    )
    source = _input(tmp_path, (*decision_events, amendment))
    result = run_golden_replay(
        input_path=source,
        output_directory=tmp_path / "output",
    )
    assert result.event_count == 9
