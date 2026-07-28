from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.shadow.iptv_p0.batch_adapter import ReferenceBatchReader

BASE_TIME = datetime(
    2026,
    1,
    2,
    12,
    0,
    tzinfo=timezone.utc,
)


def _reader(tmp_path, inventory, clock_policy, watermark=None):
    return ReferenceBatchReader(
        inventory=inventory,
        clock_policy=clock_policy,
        allowed_root=tmp_path,
        watermark_utc=watermark,
    )


def _jsonl(path, values, newline="\n"):
    payload = newline.join(json.dumps(value) for value in values) + newline
    path.write_bytes(payload.encode("utf-8"))


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_jsonl_reader_accepts_cross_platform_newlines(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
    newline,
):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="one",
    )
    path = tmp_path / "records.jsonl"
    _jsonl(path, [record.model_dump(mode="json")], newline)
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.accepted == (record,)
    assert not result.quarantine


def test_csv_reader_accepts_same_contract(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="one",
    )
    path = tmp_path / "records.csv"
    values = record.model_dump(mode="json")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(values))
        writer.writeheader()
        writer.writerow(values)
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.accepted == (record,)
    assert result.format == "csv"


@pytest.mark.parametrize(
    "payload,reason",
    (
        ("not-json\n", "MALFORMED_RECORD"),
        ("[]\n", "MALFORMED_RECORD"),
        ("\n", "MALFORMED_RECORD"),
    ),
)
def test_malformed_jsonl_is_quarantined_without_raw_content(
    tmp_path,
    inventory,
    clock_policy,
    payload,
    reason,
):
    path = tmp_path / "records.jsonl"
    path.write_text(payload, encoding="utf-8", newline="")
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.quarantine[0].reason_code == reason
    assert not result.accepted
    assert "not-json" not in result.model_dump_json()


def test_exact_duplicate_is_deduplicated(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="duplicate",
    )
    value = record.model_dump(mode="json")
    path = tmp_path / "records.jsonl"
    _jsonl(path, [value, value])
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert len(result.accepted) == 1
    assert result.exact_duplicate_count == 1


def test_identity_conflict_quarantines_all_variants(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    first = record_factory(
        metric_name="system.disk.utilization",
        sequence="conflict",
        value=1.0,
    )
    second = first.model_copy(update={"metric_value": 2.0})
    path = tmp_path / "records.jsonl"
    _jsonl(
        path,
        [
            first.model_dump(mode="json"),
            second.model_dump(mode="json"),
        ],
    )
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert not result.accepted
    assert {item.reason_code for item in result.quarantine} == {
        "IDENTITY_CONFLICT"
    }


def test_correction_is_append_only_and_auditable(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    first = record_factory(
        metric_name="system.disk.utilization",
        sequence="original",
    )
    correction = record_factory(
        metric_name="system.disk.utilization",
        sequence="correction",
        value=2.0,
        correction_of="original",
        event_time=first.event_time_utc + timedelta(seconds=1),
    )
    path = tmp_path / "records.jsonl"
    _jsonl(
        path,
        [
            first.model_dump(mode="json"),
            correction.model_dump(mode="json"),
        ],
    )
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.accepted == (first, correction)
    assert result.correction_count == 1


def test_missing_correction_target_is_quarantined(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    correction = record_factory(
        metric_name="system.disk.utilization",
        sequence="correction",
        correction_of="absent",
    )
    path = tmp_path / "records.jsonl"
    _jsonl(path, [correction.model_dump(mode="json")])
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.quarantine[0].reason_code == "CORRECTION_TARGET_MISSING"


@pytest.mark.parametrize(
    "field,value,reason",
    (
        ("source_id", "src_" + "f" * 32, "UNAPPROVED_SOURCE"),
        ("schema_version", "9.9", "SCHEMA_VERSION_MISMATCH"),
    ),
)
def test_source_identity_and_schema_conflicts_quarantine(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
    field,
    value,
    reason,
):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="one",
    )
    payload = record.model_dump(mode="json")
    payload[field] = value
    path = tmp_path / "records.jsonl"
    _jsonl(path, [payload])
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.quarantine[0].reason_code == reason


def test_record_behind_watermark_is_quarantined(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="late",
    )
    path = tmp_path / "records.jsonl"
    _jsonl(path, [record.model_dump(mode="json")])
    result = _reader(
        tmp_path,
        inventory,
        clock_policy,
        watermark=BASE_TIME,
    ).read(path)
    assert result.quarantine[0].reason_code == "AT_OR_BEHIND_WATERMARK"


def test_out_of_order_input_is_counted_and_sorted(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    later = record_factory(
        metric_name="system.disk.utilization",
        sequence="later",
        event_time=BASE_TIME - timedelta(seconds=10),
    )
    earlier = record_factory(
        metric_name="system.disk.utilization",
        sequence="earlier",
        event_time=BASE_TIME - timedelta(seconds=20),
    )
    path = tmp_path / "records.jsonl"
    _jsonl(
        path,
        [
            later.model_dump(mode="json"),
            earlier.model_dump(mode="json"),
        ],
    )
    result = _reader(tmp_path, inventory, clock_policy).read(path)
    assert result.out_of_order_count == 1
    assert result.accepted == (earlier, later)


def test_reader_is_byte_stable_across_double_run(
    tmp_path,
    inventory,
    record_factory,
    clock_policy,
):
    record = record_factory(
        metric_name="system.disk.utilization",
        sequence="one",
    )
    path = tmp_path / "records.jsonl"
    _jsonl(path, [record.model_dump(mode="json")])
    reader = _reader(tmp_path, inventory, clock_policy)
    assert reader.read(path).content_hash == reader.read(path).content_hash


def test_reader_rejects_path_relocation_escape(
    tmp_path,
    inventory,
    clock_policy,
):
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        _reader(tmp_path, inventory, clock_policy).read(outside)


def test_reader_rejects_unsupported_format(
    tmp_path,
    inventory,
    clock_policy,
):
    path = tmp_path / "records.xml"
    path.write_text("<records />", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL and CSV"):
        _reader(tmp_path, inventory, clock_policy).read(path)
