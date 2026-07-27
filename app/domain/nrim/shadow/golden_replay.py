from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import Field

from .candidate_evidence import StrictDecisionModel
from .decision_events import DecisionEvent
from .decision_projection import DecisionSnapshot, project_decision
from .hashing import bytes_hash, canonical_json, canonical_jsonl, file_hash


MAX_REPLAY_BYTES = 16 * 1024 * 1024
MAX_REPLAY_EVENTS = 10_000


class GoldenReplayResult(StrictDecisionModel):
    input_sha256: str
    events_sha256: str
    snapshots_sha256: str
    manifest_sha256: str
    event_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    output_directory: str
    compatibility: str = "frozen_v0.6_outputs_preserved"
    schema_version: str = "0.7.2"


def _safe_output(output_directory: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("Replay output filename contains traversal")
    output_directory.mkdir(parents=True, exist_ok=True)
    root = output_directory.resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError("Replay output escapes its output directory")
    return path


def load_event_jsonl(path: Path) -> tuple[DecisionEvent, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_REPLAY_BYTES:
        raise ValueError("Replay input exceeds the size limit")
    events: list[DecisionEvent] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError("Golden JSONL records must end with LF")
            if line.endswith("\r\n"):
                raise ValueError("Golden JSONL must not contain CRLF")
            if not line.strip():
                raise ValueError(f"Blank replay record at line {line_number}")
            events.append(DecisionEvent.model_validate_json(line))
            if len(events) > MAX_REPLAY_EVENTS:
                raise ValueError("Replay input exceeds the event-count limit")
    return tuple(events)


def run_golden_replay(
    *,
    input_path: Path,
    output_directory: Path,
    as_known_at_utc: datetime | None = None,
) -> GoldenReplayResult:
    events = load_event_jsonl(input_path)
    grouped: dict[str, list[DecisionEvent]] = defaultdict(list)
    for event in events:
        grouped[event.decision_id].append(event)
    ordered_events: list[DecisionEvent] = []
    snapshots: list[DecisionSnapshot] = []
    for decision_id in sorted(grouped):
        stream = tuple(
            sorted(
                grouped[decision_id],
                key=lambda item: item.stream_sequence,
            )
        )
        ordered_events.extend(stream)
        view = "as_known_at" if as_known_at_utc is not None else "latest"
        snapshot = project_decision(
            stream,
            view=view,
            as_known_at_utc=as_known_at_utc,
        )
        if not isinstance(snapshot, DecisionSnapshot):
            raise RuntimeError("Replay projection returned a comparison")
        snapshots.append(snapshot)

    events_bytes = canonical_jsonl(ordered_events)
    snapshots_bytes = canonical_jsonl(snapshots)
    events_path = _safe_output(output_directory, "events.jsonl")
    snapshots_path = _safe_output(output_directory, "snapshots.jsonl")
    manifest_path = _safe_output(output_directory, "manifest.json")
    events_path.write_bytes(events_bytes)
    snapshots_path.write_bytes(snapshots_bytes)
    manifest = {
        "schema_version": "0.7.2",
        "input_sha256": file_hash(input_path),
        "events_sha256": bytes_hash(events_bytes),
        "snapshots_sha256": bytes_hash(snapshots_bytes),
        "event_count": len(ordered_events),
        "decision_count": len(snapshots),
        "compatibility": "frozen_v0.6_outputs_preserved",
        "truth_status": "synthetic_software_evidence_only",
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return GoldenReplayResult(
        input_sha256=manifest["input_sha256"],
        events_sha256=manifest["events_sha256"],
        snapshots_sha256=manifest["snapshots_sha256"],
        manifest_sha256=bytes_hash(manifest_bytes),
        event_count=len(ordered_events),
        decision_count=len(snapshots),
        output_directory=output_directory.resolve().as_posix(),
    )


def export_events(
    events: tuple[DecisionEvent, ...],
    *,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_jsonl(events)
    output_path.write_bytes(payload)
    return bytes_hash(payload)


def seal_golden_fixture(directory: Path) -> dict[str, object]:
    required = (
        "SYNTHETIC_ONLY.md",
        "scope.json",
        "input/events.jsonl",
        "expected/events.jsonl",
        "expected/snapshots.jsonl",
        "expected/manifest.json",
        "schemas/manifest.json",
    )
    hashes: dict[str, str] = {}
    root = directory.resolve()
    for relative in required:
        path = (root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"Golden fixture file is missing: {relative}")
        hashes[relative] = file_hash(path)
    manifest = {
        "schema_version": "0.7.2",
        "files": hashes,
        "truth_status": "synthetic_software_evidence_only",
        "portable_paths_only": True,
    }
    (root / "fixture_manifest.json").write_bytes(
        (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    return manifest


def verify_golden_fixture(directory: Path) -> dict[str, object]:
    root = directory.resolve()
    manifest_path = root / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "0.7.2":
        raise ValueError("Golden fixture schema version is stale")
    if manifest.get("truth_status") != "synthetic_software_evidence_only":
        raise ValueError("Golden fixture lacks its synthetic disclaimer")
    disclaimer = (root / "SYNTHETIC_ONLY.md").read_text(encoding="utf-8")
    if "synthetic" not in disclaimer.lower() or "not production" not in (
        disclaimer.lower()
    ):
        raise ValueError("Golden fixture disclaimer is incomplete")
    for relative, expected in manifest.get("files", {}).items():
        portable = Path(relative)
        if portable.is_absolute() or ".." in portable.parts:
            raise ValueError("Golden fixture manifest path is not portable")
        path = (root / portable).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Golden fixture path escapes its directory")
        if file_hash(path) != expected:
            raise ValueError(f"Golden fixture commitment differs: {relative}")
    load_event_jsonl(root / "input/events.jsonl")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay or verify immutable NRIM decision evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--as-known-at")
    verify = subparsers.add_parser("verify-fixture")
    verify.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "replay":
        at = (
            datetime.fromisoformat(arguments.as_known_at)
            if arguments.as_known_at
            else None
        )
        result = run_golden_replay(
            input_path=arguments.input,
            output_directory=arguments.output_dir,
            as_known_at_utc=at,
        )
        print(canonical_json(result))
    else:
        print(canonical_json(verify_golden_fixture(arguments.directory)))


if __name__ == "__main__":
    main()
