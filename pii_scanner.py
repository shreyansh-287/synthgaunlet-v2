"""Deterministic PII scan and remediation. Raw values stay on the server."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import pandas as pd

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{8,15}|\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
URL_RE = re.compile(r"\bhttps?://[^\s]+", re.I)
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
NAME_HINTS = ("name", "first_name", "last_name", "fullname", "full_name")
ADDRESS_HINTS = ("address", "street", "city", "zip", "postal")
ID_HINTS = ("ssn", "social", "dob", "date_of_birth", "birth", "passport", "account", "customer_id", "user_id")

ACTIONS = {"mask", "drop", "hash", "faker"}


def scan(frame: pd.DataFrame) -> list[dict]:
    findings = []
    for column in frame.columns:
        kind, count = _column_finding(frame[column], str(column))
        if kind and count:
            findings.append({
                "column": str(column),
                "kind": kind,
                "count": int(count),
                "redacted_example": _redact(frame[column].dropna().astype(str).head(1).tolist()),
            })
    return findings


def apply(frame: pd.DataFrame, actions: Iterable[dict], salt: str = "synthgauntlet") -> pd.DataFrame:
    result = frame.copy()
    faker = None
    for item in actions:
        column = str(item.get("column", ""))
        action = str(item.get("action", "")).lower()
        if column not in result.columns or action not in ACTIONS:
            continue
        if action == "drop":
            result = result.drop(columns=[column])
            continue
        if action == "mask":
            result[column] = "[REDACTED]"
        elif action == "hash":
            result[column] = result[column].map(lambda value: _hash(value, salt))
        else:
            if faker is None:
                from faker import Faker
                faker = Faker()
                faker.seed_instance(42)
            result[column] = [_fake(faker, column, index) for index in range(len(result))]
    return result


def _column_finding(series: pd.Series, name: str) -> tuple[str, int]:
    values = series.dropna().astype(str)
    checks = (
        ("email", EMAIL_RE),
        ("ssn", SSN_RE),
        ("credit_card", None),
        ("phone", PHONE_RE),
        ("ip_address", IP_RE),
        ("url", URL_RE),
    )
    for kind, pattern in checks:
        if kind == "credit_card":
            count = int(values.map(_is_card).sum())
        else:
            count = int(values.map(lambda value, pattern=pattern: bool(pattern.search(value))).sum())
        if count:
            return kind, count

    lowered = name.lower()
    if any(hint in lowered for hint in ("email", *NAME_HINTS, *ADDRESS_HINTS, *ID_HINTS, "phone")):
        return "sensitive_column_name", int(len(values))
    if "id" in lowered and len(values) >= 20 and values.nunique() / max(len(values), 1) > 0.95:
        return "surrogate_id", int(len(values))
    return "", 0


def _is_card(value: str) -> bool:
    match = CARD_RE.search(value)
    if not match:
        return False
    digits = re.sub(r"\D", "", match.group(0))
    if not 13 <= len(digits) <= 19 or SSN_RE.fullmatch(value.strip()):
        return False
    return _luhn(digits)


def _luhn(digits: str) -> bool:
    total = 0
    reverse = digits[::-1]
    for index, character in enumerate(reverse):
        number = int(character)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _redact(samples: list[str]) -> str:
    if not samples:
        return ""
    value = samples[0]
    if "@" in value:
        name, _, domain = value.partition("@")
        return f"{name[:1]}***@{domain}"
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return f"***{digits[-4:]}"
    return f"{value[:1]}***" if value else ""


def _hash(value, salt: str) -> str:
    raw = "" if pd.isna(value) else str(value)
    return hashlib.sha256(f"{salt}:{raw}".encode()).hexdigest()[:16]


def _fake(faker, column: str, index: int) -> str:
    lowered = column.lower()
    if "email" in lowered:
        return faker.email()
    if "phone" in lowered:
        return faker.phone_number()
    if any(hint in lowered for hint in NAME_HINTS):
        return faker.name()
    if any(hint in lowered for hint in ADDRESS_HINTS):
        return faker.address().replace("\n", ", ")
    return faker.bothify(text=f"synthetic-{index}-????")
