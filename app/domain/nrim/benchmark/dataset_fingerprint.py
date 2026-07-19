from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import DatasetFingerprint


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(
    value: Any,
) -> bytes:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return serialized.encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        canonical_json_bytes(value)
    )


def sha256_file(path: str | Path) -> str:
    resolved_path = Path(path)

    digest = hashlib.sha256()

    with resolved_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def build_dataset_fingerprint(
    *,
    manifest: dict[str, Any],
) -> DatasetFingerprint:
    records = sorted(
        manifest["records"],
        key=lambda item: str(item["window_id"]),
    )

    file_hashes: dict[str, str] = {}

    for record in records:
        window_id = str(record["window_id"])
        path = Path(str(record["path"]))

        file_hashes[window_id] = sha256_file(path)

    manifest_hash = sha256_json(
        {
            "dataset_name": manifest.get(
                "dataset_name"
            ),
            "dataset_version": manifest.get(
                "dataset_version"
            ),
            "source_dataset_version": manifest.get(
                "source_dataset_version"
            ),
            "window_specification": manifest.get(
                "window_specification"
            ),
            "records": records,
        }
    )

    feature_schema_hash = sha256_json(
        manifest["feature_schema"]
    )

    ordered_window_hash = sha256_json(
        [
            {
                "window_id": window_id,
                "sha256": file_hashes[window_id],
            }
            for window_id in sorted(file_hashes)
        ]
    )

    benchmark_hash = sha256_json(
        {
            "manifest_sha256": manifest_hash,
            "feature_schema_sha256": (
                feature_schema_hash
            ),
            "ordered_window_hash_sha256": (
                ordered_window_hash
            ),
        }
    )

    return DatasetFingerprint(
        dataset_manifest_sha256=manifest_hash,
        feature_schema_sha256=feature_schema_hash,
        ordered_window_hash_sha256=ordered_window_hash,
        benchmark_sha256=benchmark_hash,
        window_file_hashes=file_hashes,
    )


def verify_dataset_fingerprint(
    *,
    manifest: dict[str, Any],
    expected: DatasetFingerprint,
) -> list[str]:
    actual = build_dataset_fingerprint(
        manifest=manifest
    )

    errors: list[str] = []

    if actual != expected:
        if (
            actual.dataset_manifest_sha256
            != expected.dataset_manifest_sha256
        ):
            errors.append(
                "Dataset manifest fingerprint changed."
            )

        if (
            actual.feature_schema_sha256
            != expected.feature_schema_sha256
        ):
            errors.append(
                "Feature schema fingerprint changed."
            )

        if (
            actual.ordered_window_hash_sha256
            != expected.ordered_window_hash_sha256
        ):
            errors.append(
                "One or more window files changed."
            )

        if (
            actual.benchmark_sha256
            != expected.benchmark_sha256
        ):
            errors.append(
                "Overall benchmark fingerprint changed."
            )

    return errors
