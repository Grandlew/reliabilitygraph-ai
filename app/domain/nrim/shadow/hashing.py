from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _canonical_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TypeError("Canonical datetimes must include a timezone")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(
        f"Unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used for evidence hashes."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_canonical_default,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def typed_hash(*, type_name: str, schema_version: str, value: Any) -> str:
    """Hash one typed/versioned value to prevent cross-contract collisions."""

    return canonical_hash(
        {
            "schema_version": schema_version,
            "type": type_name,
            "value": value,
        }
    )


def canonical_jsonl(values: list[Any] | tuple[Any, ...]) -> bytes:
    return b"".join(
        canonical_json(value).encode("utf-8") + b"\n"
        for value in values
    )


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
