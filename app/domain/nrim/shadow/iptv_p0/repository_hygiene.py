from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..hashing import canonical_hash


_FORBIDDEN = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ipv4": re.compile(
        r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"
    ),
    "mac": re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    "credential": re.compile(
        r"(?i)\b(?:password|api[_-]?key|secret|bearer)\s*[:=]\s*\S+"
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


class HygieneFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relative_path: str
    rule: str
    line_number: int
    line_sha256: str


def scan_fixture_tree(root: Path) -> tuple[HygieneFinding, ...]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    marker = resolved / "SYNTHETIC_ONLY.md"
    if not marker.is_file():
        raise ValueError("Synthetic fixture marker is required")
    findings = []
    for path in sorted(resolved.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".json",
            ".jsonl",
            ".csv",
            ".md",
        }:
            continue
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Fixture file exceeds hygiene scan limit")
        with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
            for line_number, line in enumerate(stream, start=1):
                for rule, pattern in _FORBIDDEN.items():
                    if pattern.search(line):
                        findings.append(
                            HygieneFinding(
                                relative_path=path.relative_to(
                                    resolved
                                ).as_posix(),
                                rule=rule,
                                line_number=line_number,
                                line_sha256=canonical_hash(
                                    {"line": line.rstrip("\r\n")}
                                ),
                            )
                        )
    return tuple(findings)
