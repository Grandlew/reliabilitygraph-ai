from __future__ import annotations

import json

import pytest

from app.domain.nrim.shadow.iptv_p0.topology import load_topology_jsonl


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_offline_topology_reader_accepts_cross_platform_jsonl(
    tmp_path,
    topology_capture,
    newline,
):
    path = tmp_path / "topology.jsonl"
    path.write_bytes(
        (
            json.dumps(topology_capture.model_dump(mode="json")) + newline
        ).encode("utf-8")
    )
    history = load_topology_jsonl(path, allowed_root=tmp_path)
    result = history.reconstruct(
        event_cutoff_utc=topology_capture.effective_at_utc,
        knowledge_cutoff_utc=topology_capture.recorded_at_utc,
    )
    assert result.topology_version == topology_capture.topology_version


@pytest.mark.parametrize("payload", ("not-json\n", "[]\n", "\n"))
def test_offline_topology_reader_rejects_malformed_lines(tmp_path, payload):
    path = tmp_path / "topology.jsonl"
    path.write_text(payload, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="line 1"):
        load_topology_jsonl(path, allowed_root=tmp_path)


def test_offline_topology_reader_rejects_path_escape(tmp_path):
    path = tmp_path.parent / "outside-topology.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_topology_jsonl(path, allowed_root=tmp_path)


def test_offline_topology_reader_rejects_non_jsonl(
    tmp_path,
    topology_capture,
):
    path = tmp_path / "topology.json"
    path.write_text(
        json.dumps(topology_capture.model_dump(mode="json")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JSONL"):
        load_topology_jsonl(path, allowed_root=tmp_path)
