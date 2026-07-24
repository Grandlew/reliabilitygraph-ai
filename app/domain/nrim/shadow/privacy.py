from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping, Sequence
from typing import Any


DIRECT_IDENTIFIER_KEYS = {
    "customer",
    "customer_id",
    "customer_name",
    "deployment_id",
    "deployment_name",
    "employee",
    "employee_id",
    "email",
    "first_name",
    "full_name",
    "hotel",
    "hotel_name",
    "last_name",
    "phone",
    "room",
    "room_number",
    "subscriber",
    "subscriber_id",
    "username",
}

FORBIDDEN_INFERENCE_TOKENS = {
    "adjudication",
    "confounder",
    "customer",
    "diagnosis",
    "failure_type",
    "fault_id",
    "future",
    "hidden",
    "label",
    "pair_id",
    "remediation",
    "resolution",
    "reviewer",
    "scenario_id",
    "ticket",
}

_EMAIL = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
_PHONE = re.compile(
    r"(?<!\w)(?:"
    r"\+\d{10,15}"
    r"|\(\d{2,4}\)[\d -]{6,}\d"
    r"|\d{3}[- ]\d{3}[- ]\d{4}"
    r")(?!\w)"
)


class PrivacyViolation(ValueError):
    pass


class Pseudonymizer:
    """Domain-separated HMAC pseudonyms.

    A keyed construction prevents dictionary reversal of predictable deployment
    or component names. The secret must be supplied by the deployment secret
    manager and is never stored in evidence or model artifacts.
    """

    def __init__(self, secret: bytes, *, key_id: str) -> None:
        if len(secret) < 32:
            raise ValueError("Pseudonymization secret must be at least 32 bytes")
        if not key_id.strip():
            raise ValueError("Pseudonymization key_id is required")
        self._secret = bytes(secret)
        self.key_id = key_id

    def pseudonymize(self, value: str, *, namespace: str) -> str:
        clean_value = value.strip()
        clean_namespace = namespace.strip().lower()
        if not clean_value or not clean_namespace:
            raise ValueError("Pseudonym value and namespace are required")
        digest = hmac.new(
            self._secret,
            f"{clean_namespace}\0{clean_value}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{clean_namespace}_{digest[:40]}"


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def assert_no_direct_identifiers(value: Any) -> None:
    findings: list[str] = []
    for path, child in _walk(value):
        key = path[-1].lower() if path else ""
        if key in DIRECT_IDENTIFIER_KEYS:
            findings.append(".".join(path))
        if isinstance(child, str):
            location = ".".join(path) or "<root>"
            if _EMAIL.search(child):
                findings.append(location + "<email>")
            if _PHONE.search(child):
                findings.append(location + "<phone>")
    if findings:
        raise PrivacyViolation(
            "Direct identifier material found at: "
            + ", ".join(sorted(set(findings)))
        )


def assert_inference_payload_is_label_free(value: Any) -> None:
    findings = []
    for path, _ in _walk(value):
        key = path[-1].lower() if path else ""
        if any(token in key for token in FORBIDDEN_INFERENCE_TOKENS):
            findings.append(".".join(path))
    if findings:
        raise PrivacyViolation(
            "Forbidden inference fields found at: "
            + ", ".join(sorted(set(findings)))
        )


def redacted_structure(value: Any) -> Any:
    """Retain safe structure for quarantine while dropping sensitive values."""

    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in DIRECT_IDENTIFIER_KEYS:
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = redacted_structure(child)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [redacted_structure(item) for item in value]
    if isinstance(value, str):
        if _EMAIL.search(value) or _PHONE.search(value):
            return "<redacted>"
        return value[:256]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"
