"""D1 query helper + patient resolution (handles Synthea numeric suffixes)."""

from __future__ import annotations

import re
from typing import Any

import requests

from .config import D1_API_URL


class D1Error(RuntimeError):
    pass


class PatientNotFound(LookupError):
    pass


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def query(sql: str, *, timeout: int = 15) -> list[dict[str, Any]]:
    """Run a SELECT against the public D1 worker. Returns the results list."""
    try:
        resp = requests.post(D1_API_URL, json={"sql": sql}, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
    except requests.RequestException as exc:
        raise D1Error(f"D1 request failed: {exc}") from exc

    if not body.get("success", False):
        raise D1Error(f"D1 error: {body.get('error', body)}")
    return body.get("results", [])


def _escape(value: str) -> str:
    return value.replace("'", "''")


def resolve_patient(input_str: str) -> dict[str, Any]:
    """Resolve a UUID or fuzzy name into a patient_summary row.

    Synthea names carry numeric suffixes ("Lindsay928 Brekke496"), so we
    use case-insensitive LIKE on tokens.
    """
    s = input_str.strip()
    if _UUID_RE.match(s):
        rows = query(f"SELECT * FROM patient_summary WHERE id = '{_escape(s)}' LIMIT 1")
        if not rows:
            raise PatientNotFound(f"No patient with id={s}")
        return rows[0]

    tokens = [t for t in re.split(r"\s+", s) if t]
    if not tokens:
        raise PatientNotFound("Empty name")

    if len(tokens) == 1:
        t = _escape(tokens[0].lower())
        where = (
            f"(LOWER(first) LIKE '%{t}%' OR LOWER(last) LIKE '%{t}%')"
        )
    else:
        first_t = _escape(tokens[0].lower())
        last_t = _escape(tokens[-1].lower())
        where = (
            f"LOWER(first) LIKE '%{first_t}%' AND LOWER(last) LIKE '%{last_t}%'"
        )

    rows = query(
        f"SELECT * FROM patient_summary WHERE {where} "
        "ORDER BY ed_inpatient_total_cost DESC LIMIT 5"
    )
    if not rows:
        raise PatientNotFound(f"No patient matching '{input_str}'")
    return rows[0]
