from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from app.domain.nrim.shadow.hashing import (
    canonical_hash,
    canonical_json,
    canonical_jsonl,
    typed_hash,
)


@pytest.mark.parametrize("unsafe_value", [b"opaque", {"unordered", "set"}])
def test_canonical_json_rejects_unsafe_process_dependent_values(
    unsafe_value: object,
) -> None:
    with pytest.raises(TypeError):
        canonical_json({"value": unsafe_value})


def test_canonical_hash_ignores_dictionary_insertion_order() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash(
        {"b": 2, "a": 1}
    )


def test_typed_hash_prevents_cross_contract_collisions() -> None:
    value = {"same": "payload"}
    assert typed_hash(
        type_name="candidate_set",
        schema_version="0.7.2",
        value=value,
    ) != typed_hash(
        type_name="causal_evidence_set",
        schema_version="0.7.2",
        value=value,
    )


def test_typed_hash_binds_schema_version() -> None:
    assert typed_hash(
        type_name="decision",
        schema_version="0.7.1",
        value={"value": 1},
    ) != typed_hash(
        type_name="decision",
        schema_version="0.7.2",
        value={"value": 1},
    )


def test_canonical_json_normalizes_utc_offsets() -> None:
    instant = datetime.fromisoformat("2026-07-25T13:00:00+03:00")
    assert canonical_json({"time": instant}) == (
        '{"time":"2026-07-25T10:00:00+00:00"}'
    )


def test_canonical_json_rejects_naive_datetime() -> None:
    with pytest.raises(TypeError):
        canonical_json({"time": datetime(2026, 7, 25)})


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})


def test_canonical_jsonl_is_lf_terminated() -> None:
    payload = canonical_jsonl(({"b": 2, "a": 1}, {"value": "x"}))
    assert payload == b'{"a":1,"b":2}\n{"value":"x"}\n'
    assert b"\r\n" not in payload


def test_semantic_mutation_changes_commitment() -> None:
    assert canonical_hash({"state": "unknown"}) != canonical_hash(
        {"state": "healthy"}
    )


def test_hash_is_stable_in_a_fresh_process() -> None:
    expected = canonical_hash({"type": "decision", "value": [1, 2, 3]})
    script = (
        "from app.domain.nrim.shadow.hashing import canonical_hash;"
        "print(canonical_hash({'value':[1,2,3],'type':'decision'}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == expected


def test_canonical_serialization_round_trips_json() -> None:
    encoded = canonical_json({"time": datetime(2026, 7, 25, tzinfo=timezone.utc)})
    assert json.loads(encoded) == {"time": "2026-07-25T00:00:00+00:00"}
