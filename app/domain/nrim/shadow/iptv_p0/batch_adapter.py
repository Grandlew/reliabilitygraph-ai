from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..hashing import canonical_hash
from ..privacy import assert_no_direct_identifiers
from .clock_watermark import (
    ClockWatermarkPolicy,
    TimeDisposition,
    assess_record_time,
)
from .contracts import utc
from .source_inventory import SourceInventory
from .signal_registry import AvailabilityClass, SignalRegistry


MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_BATCH_RECORDS = 1_000_000


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class BatchInputRecord(StrictModel):
    source_id: str = Field(pattern=r"^src_[0-9a-f]{32}$")
    source_sequence_id: str = Field(min_length=1, max_length=256)
    schema_version: str = Field(min_length=1, max_length=64)
    metric_name: str = Field(min_length=3, max_length=256)
    metric_value: float | None
    unit: str = Field(min_length=1, max_length=128)
    component_pseudonym: str = Field(
        min_length=12,
        max_length=128,
        pattern=r"^cmp_[0-9a-f]{8,64}$",
    )
    event_time_utc: datetime
    observation_time_utc: datetime
    ingestion_time_utc: datetime
    quality: str = Field(min_length=2, max_length=64)
    applicability: str = Field(min_length=2, max_length=64)
    correction_of: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )

    @field_validator(
        "event_time_utc",
        "observation_time_utc",
        "ingestion_time_utc",
    )
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def validate_value_state(self) -> Self:
        if self.applicability == "observed":
            if self.metric_value is None or not math.isfinite(
                self.metric_value
            ):
                raise ValueError("Observed records require a finite value")
        elif self.metric_value is not None:
            raise ValueError("Absent/inapplicable records require null value")
        assert_no_direct_identifiers(self.model_dump(mode="json"))
        return self

    @property
    def record_key(self) -> str:
        return f"{self.source_id}:{self.source_sequence_id}"

    @property
    def record_sha256(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class QuarantineRecord(StrictModel):
    line_number: int = Field(gt=0)
    reason_code: str = Field(min_length=3, max_length=128)
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_key: str | None = None


class BatchAdapterResult(StrictModel):
    source_path_name: str
    format: str
    accepted: tuple[BatchInputRecord, ...]
    quarantine: tuple[QuarantineRecord, ...]
    exact_duplicate_count: int = Field(ge=0)
    correction_count: int = Field(ge=0)
    out_of_order_count: int = Field(ge=0)
    input_record_count: int = Field(ge=0)

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class OfflineBatchAdapter(Protocol):
    def read(self, path: Path) -> BatchAdapterResult: ...


class ReferenceBatchReader:
    """Deterministic, offline-only JSONL/CSV reader."""

    def __init__(
        self,
        *,
        inventory: SourceInventory,
        clock_policy: ClockWatermarkPolicy,
        allowed_root: Path,
        watermark_utc: datetime | None = None,
        signal_registry: SignalRegistry | None = None,
    ) -> None:
        self.inventory = inventory
        self.clock_policy = clock_policy
        self.allowed_root = allowed_root.resolve()
        self.watermark_utc = watermark_utc
        self.signal_registry = signal_registry
        self._sources = {
            item.source_id: item for item in inventory.sources
        }

    def _resolve(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.allowed_root):
            raise ValueError("Batch input escapes the approved offline root")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved.stat().st_size > MAX_BATCH_BYTES:
            raise ValueError("Batch input exceeds the size limit")
        if resolved.suffix.lower() not in {".jsonl", ".csv"}:
            raise ValueError("Only offline JSONL and CSV are supported")
        return resolved

    @staticmethod
    def _jsonl(path: Path) -> list[tuple[int, dict[str, Any] | None, str]]:
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                raw_hash = canonical_hash({"raw_line": line})
                if not line.strip():
                    rows.append((line_number, None, raw_hash))
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    rows.append((line_number, None, raw_hash))
                    continue
                rows.append(
                    (
                        line_number,
                        value if isinstance(value, dict) else None,
                        raw_hash,
                    )
                )
        return rows

    @staticmethod
    def _csv(path: Path) -> list[tuple[int, dict[str, Any] | None, str]]:
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for line_number, value in enumerate(reader, start=2):
                normalized: dict[str, Any] = dict(value)
                raw_value = normalized.get("metric_value")
                normalized["metric_value"] = (
                    None if raw_value in {None, "", "null"} else raw_value
                )
                correction = normalized.get("correction_of")
                normalized["correction_of"] = correction or None
                rows.append(
                    (
                        line_number,
                        normalized,
                        canonical_hash({"csv_row": dict(value)}),
                    )
                )
        return rows

    def read(self, path: Path) -> BatchAdapterResult:
        resolved = self._resolve(path)
        raw_rows = (
            self._jsonl(resolved)
            if resolved.suffix.lower() == ".jsonl"
            else self._csv(resolved)
        )
        if len(raw_rows) > MAX_BATCH_RECORDS:
            raise ValueError("Batch input exceeds the record limit")

        parsed: list[tuple[int, BatchInputRecord]] = []
        quarantine: list[QuarantineRecord] = []
        for line_number, raw, raw_hash in raw_rows:
            if raw is None:
                quarantine.append(
                    QuarantineRecord(
                        line_number=line_number,
                        reason_code="MALFORMED_RECORD",
                        source_record_sha256=raw_hash,
                    )
                )
                continue
            try:
                record = BatchInputRecord.model_validate(raw)
            except ValidationError:
                quarantine.append(
                    QuarantineRecord(
                        line_number=line_number,
                        reason_code="CONTRACT_INVALID",
                        source_record_sha256=raw_hash,
                    )
                )
                continue
            source = self._sources.get(record.source_id)
            if source is None:
                quarantine.append(
                    QuarantineRecord(
                        line_number=line_number,
                        reason_code="UNAPPROVED_SOURCE",
                        source_record_sha256=record.record_sha256,
                        record_key=record.record_key,
                    )
                )
                continue
            if record.schema_version != source.schema_version:
                quarantine.append(
                    QuarantineRecord(
                        line_number=line_number,
                        reason_code="SCHEMA_VERSION_MISMATCH",
                        source_record_sha256=record.record_sha256,
                        record_key=record.record_key,
                    )
                )
                continue
            if self.signal_registry is not None:
                try:
                    validate_profile_semantics(
                        record,
                        registry=self.signal_registry,
                    )
                except ValueError:
                    quarantine.append(
                        QuarantineRecord(
                            line_number=line_number,
                            reason_code="PROFILE_SEMANTIC_MISMATCH",
                            source_record_sha256=record.record_sha256,
                            record_key=record.record_key,
                        )
                    )
                    continue
            time = assess_record_time(
                event_time_utc=record.event_time_utc,
                observation_time_utc=record.observation_time_utc,
                ingestion_time_utc=record.ingestion_time_utc,
                clock_quality=source.clock_quality,
                policy=self.clock_policy,
                watermark_utc=self.watermark_utc,
            )
            if time.disposition is not TimeDisposition.ACCEPT:
                quarantine.append(
                    QuarantineRecord(
                        line_number=line_number,
                        reason_code=time.reason_code,
                        source_record_sha256=record.record_sha256,
                        record_key=record.record_key,
                    )
                )
                continue
            parsed.append((line_number, record))

        grouped: dict[str, list[tuple[int, BatchInputRecord]]] = defaultdict(
            list
        )
        for item in parsed:
            grouped[item[1].record_key].append(item)

        accepted: list[tuple[int, BatchInputRecord]] = []
        duplicates = 0
        for key in sorted(grouped):
            group = grouped[key]
            distinct = {record.record_sha256 for _, record in group}
            if len(distinct) > 1:
                for line_number, record in group:
                    quarantine.append(
                        QuarantineRecord(
                            line_number=line_number,
                            reason_code="IDENTITY_CONFLICT",
                            source_record_sha256=record.record_sha256,
                            record_key=record.record_key,
                        )
                    )
                continue
            accepted.append(min(group, key=lambda item: item[0]))
            duplicates += len(group) - 1

        accepted.sort(key=lambda item: item[0])
        accepted_keys = {record.record_key for _, record in accepted}
        valid: list[tuple[int, BatchInputRecord]] = []
        corrections = 0
        for line_number, record in accepted:
            if record.correction_of is not None:
                target = f"{record.source_id}:{record.correction_of}"
                if target not in accepted_keys:
                    quarantine.append(
                        QuarantineRecord(
                            line_number=line_number,
                            reason_code="CORRECTION_TARGET_MISSING",
                            source_record_sha256=record.record_sha256,
                            record_key=record.record_key,
                        )
                    )
                    continue
                corrections += 1
            valid.append((line_number, record))

        original_order = [record.event_time_utc for _, record in valid]
        out_of_order = sum(
            left > right
            for left, right in zip(
                original_order,
                original_order[1:],
                strict=False,
            )
        )
        ordered = tuple(
            record
            for _, record in sorted(
                valid,
                key=lambda item: (
                    item[1].event_time_utc,
                    item[1].source_id,
                    item[1].source_sequence_id,
                    item[0],
                ),
            )
        )
        return BatchAdapterResult(
            source_path_name=resolved.name,
            format=resolved.suffix.lower().lstrip("."),
            accepted=ordered,
            quarantine=tuple(
                sorted(
                    quarantine,
                    key=lambda item: (
                        item.line_number,
                        item.reason_code,
                        item.source_record_sha256,
                    ),
                )
            ),
            exact_duplicate_count=duplicates,
            correction_count=corrections,
            out_of_order_count=out_of_order,
            input_record_count=len(raw_rows),
        )


def validate_profile_semantics(
    record: BatchInputRecord,
    *,
    registry: SignalRegistry,
) -> None:
    try:
        requirement = registry.requirement(record.metric_name)
    except KeyError as error:
        raise ValueError("Signal is not in the frozen registry") from error
    if requirement.availability is AvailabilityClass.UNAVAILABLE:
        raise ValueError("Unavailable signals cannot have observations")
    if (
        requirement.source_ids
        and record.source_id not in requirement.source_ids
    ):
        raise ValueError("Signal observation came from an unapproved source")
    if record.unit != requirement.canonical_unit:
        raise ValueError("Signal unit differs from the deployment profile")
    if record.applicability not in {
        "observed",
        "missing",
        "not_applicable",
    }:
        raise ValueError("Signal applicability state is unknown")
    if record.metric_value is not None:
        if record.metric_value < 0:
            raise ValueError("Signal values cannot be negative")
        if (
            requirement.canonical_unit == "percent"
            and record.metric_value > 100
        ):
            raise ValueError("Percent signal is outside [0, 100]")
