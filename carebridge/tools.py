"""Data-pulling tools. Each function runs deterministic SQL and returns dicts.

The same functions are exposed to the Groq agent via JSON schemas so the LLM
can fetch additional context if the canned profile is not enough.
"""

from __future__ import annotations

from typing import Any

from .db import query

OPIOID_KEYWORDS = (
    "opioid",
    "hydrocodone",
    "oxycodone",
    "fentanyl",
    "morphine",
    "tramadol",
    "codeine",
)

SDOH_KEYWORDS = (
    "stress",
    "employ",
    "housing",
    "homeless",
    "transport",
    "food",
    "social contact",
    "refugee",
    "abuse",
    "education",
    "criminal",
)

DUMMY_RESOURCES = {
    "financial": [
        {"name": "Local Charity Care Program", "link": "https://example.com/charity-care"},
        {"name": "Patient Advocate Foundation", "link": "https://example.com/patient-advocate"}
    ],
    "housing": [
        {"name": "City Housing Authority", "link": "https://example.com/housing"},
        {"name": "Emergency Shelter Hotline", "link": "https://example.com/shelter"}
    ],
    "transportation": [
        {"name": "Non-Emergency Medical Transport", "link": "https://example.com/nemt"},
        {"name": "Community Ride Share", "link": "https://example.com/rideshare"}
    ],
    "food": [
        {"name": "Meals on Wheels", "link": "https://example.com/meals"},
        {"name": "Local Food Bank Directory", "link": "https://example.com/foodbank"}
    ],
    "employment": [
        {"name": "State Workforce Commission", "link": "https://example.com/workforce"},
        {"name": "Job Training Program", "link": "https://example.com/job-training"}
    ],
    "social": [
        {"name": "Community Senior Center", "link": "https://example.com/senior-center"},
        {"name": "Support Group Network", "link": "https://example.com/support-group"}
    ],
    "polypharmacy": [
        {"name": "Clinical Pharmacist Consult", "link": "https://example.com/pharm-consult"}
    ],
    "opioids": [
        {"name": "Pain Management Clinic", "link": "https://example.com/pain-management"},
        {"name": "Substance Use Support", "link": "https://example.com/substance-use"}
    ]
}

PRAPARE_QUESTIONS = (
    "Housing status",
    "Are you worried about losing your housing?",
    "Do you have a car or can you get a ride in a car?",
    "What is your current work situation?",
    "Stress level",
    "How often do you see or talk to people that you care about and feel close to?",
    "How hard is it for you to pay for the very basics like food, housing, medical care, and heating?",
)

SCREENING_PATTERNS = {
    "depression": "%depression%screening%",
    "substance_use": "%substance%use%",
    "medication_reconciliation": "%medication reconciliation%",
    "annual_wellness": "%wellness%",
    "alcohol": "%alcohol%use%",
}


def _esc(s: str) -> str:
    return s.replace("'", "''")


def pull_demographics(patient_id: str) -> dict[str, Any]:
    """One row from patient_summary plus income/lat/lon from patients."""
    pid = _esc(patient_id)
    summary = query(f"SELECT * FROM patient_summary WHERE id = '{pid}' LIMIT 1")
    extras = query(
        "SELECT INCOME, LAT, LON, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE "
        f"FROM patients WHERE Id = '{pid}' LIMIT 1"
    )
    if not summary:
        return {"found": False}
    row = summary[0]
    if extras:
        row.update({k.lower(): v for k, v in extras[0].items()})
    return {"found": True, **row}


def pull_clinical(patient_id: str) -> dict[str, Any]:
    """Active conditions, medications (with flags), care plan status."""
    pid = _esc(patient_id)

    conditions = query(
        "SELECT START, DESCRIPTION, CODE FROM conditions "
        f"WHERE PATIENT = '{pid}' AND STOP IS NULL "
        "ORDER BY START DESC LIMIT 200"
    )

    meds = query(
        "SELECT START, DESCRIPTION, CODE, TOTALCOST, DISPENSES, REASONDESCRIPTION "
        f"FROM medications WHERE PATIENT = '{pid}' AND STOP IS NULL "
        "ORDER BY START DESC LIMIT 200"
    )
    for m in meds:
        desc = (m.get("DESCRIPTION") or "").lower()
        m["is_opioid"] = any(k in desc for k in OPIOID_KEYWORDS)

    careplans = query(
        "SELECT START, DESCRIPTION, REASONDESCRIPTION FROM careplans "
        f"WHERE PATIENT = '{pid}' AND STOP IS NULL "
        "ORDER BY START DESC LIMIT 50"
    )

    return {
        "active_conditions": conditions,
        "active_medications": meds,
        "active_care_plans": careplans,
        "active_med_count": len(meds),
        "active_opioid_count": sum(1 for m in meds if m["is_opioid"]),
        "has_active_careplan": len(careplans) > 0,
    }


def pull_sdoh(patient_id: str) -> dict[str, Any]:
    """SDOH conditions + most-recent PRAPARE responses."""
    pid = _esc(patient_id)

    sdoh_clauses = " OR ".join(
        [f"LOWER(DESCRIPTION) LIKE '%{kw}%'" for kw in SDOH_KEYWORDS]
    )
    sdoh = query(
        "SELECT START, STOP, DESCRIPTION, CODE FROM conditions "
        f"WHERE PATIENT = '{pid}' AND ({sdoh_clauses}) "
        "ORDER BY START DESC LIMIT 100"
    )

    in_list = ", ".join(f"'{_esc(q)}'" for q in PRAPARE_QUESTIONS)
    obs_rows = query(
        "SELECT DATE, DESCRIPTION, VALUE, UNITS FROM observations "
        f"WHERE PATIENT = '{pid}' AND DESCRIPTION IN ({in_list}) "
        "ORDER BY DATE DESC LIMIT 200"
    )
    prapare: dict[str, dict[str, Any]] = {}
    for r in obs_rows:
        desc = r.get("DESCRIPTION")
        if desc and desc not in prapare:
            prapare[desc] = {"value": r.get("VALUE"), "date": r.get("DATE")}

    return {"sdoh_conditions": sdoh, "prapare": prapare}


def pull_debt(patient_id: str) -> dict[str, Any]:
    """Outstanding balances from claims_transactions (uses PATIENTID, not PATIENT)."""
    pid = _esc(patient_id)
    rows = query(
        "SELECT SUM(OUTSTANDING) AS total_outstanding, "
        "COUNT(*) AS unpaid_lines, "
        "MIN(FROMDATE) AS oldest_unpaid_date, "
        "MAX(FROMDATE) AS newest_unpaid_date "
        f"FROM claims_transactions WHERE PATIENTID = '{pid}' AND OUTSTANDING > 0"
    )
    r = rows[0] if rows else {}
    total = r.get("total_outstanding") or 0
    return {
        "total_outstanding": float(total or 0),
        "unpaid_lines": int(r.get("unpaid_lines") or 0),
        "oldest_unpaid_date": r.get("oldest_unpaid_date"),
        "newest_unpaid_date": r.get("newest_unpaid_date"),
    }


def pull_encounters(patient_id: str, *, limit: int = 25) -> dict[str, Any]:
    """Most-recent acute encounters (ED, inpatient, urgent care) with cost and reason.

    Synthea data is historical, so we use a fixed `limit` of recent rows
    rather than a date-relative window.
    """
    pid = _esc(patient_id)
    rows = query(
        "SELECT START, ENCOUNTERCLASS, DESCRIPTION, REASONDESCRIPTION, "
        "TOTAL_CLAIM_COST FROM encounters "
        f"WHERE PATIENT = '{pid}' "
        "AND ENCOUNTERCLASS IN ('emergency','inpatient','urgentcare') "
        f"ORDER BY START DESC LIMIT {int(limit)}"
    )
    ed = [r for r in rows if r.get("ENCOUNTERCLASS") == "emergency"]
    ip = [r for r in rows if r.get("ENCOUNTERCLASS") == "inpatient"]
    return {
        "recent_acute_encounters": rows,
        "ed_count_recent": len(ed),
        "inpatient_count_recent": len(ip),
        "total_acute_cost_recent": sum(
            float(r.get("TOTAL_CLAIM_COST") or 0) for r in rows
        ),
    }


def pull_care_gaps(patient_id: str) -> dict[str, Any]:
    """Detect missing screenings/preventive care."""
    pid = _esc(patient_id)
    gaps = {}
    for label, pattern in SCREENING_PATTERNS.items():
        rows = query(
            "SELECT COUNT(*) AS n FROM procedures "
            f"WHERE PATIENT = '{pid}' AND LOWER(DESCRIPTION) LIKE '{_esc(pattern.lower())}'"
        )
        n = int((rows[0].get("n") if rows else 0) or 0)
        gaps[label] = {"present": n > 0, "count": n}
    return {"screenings": gaps}


def scan_high_barrier_patients(top_n: int = 10) -> list[dict[str, Any]]:
    """Rank cohort by composite barrier score.

    Score = ed_visits * 2 + chronic_condition_count + (no care plan)*3
            + (debt > $1000)*2
    """
    rows = query(
        "SELECT ps.id, ps.first, ps.last, ps.ed_visits, ps.chronic_condition_count, "
        "ps.has_active_careplan, ps.ed_inpatient_total_cost, "
        "COALESCE(d.total_outstanding, 0) AS total_outstanding "
        "FROM patient_summary ps "
        "LEFT JOIN ("
        "  SELECT PATIENTID AS id, SUM(OUTSTANDING) AS total_outstanding "
        "  FROM claims_transactions WHERE OUTSTANDING > 0 GROUP BY PATIENTID"
        ") d ON d.id = ps.id "
        f"ORDER BY ps.ed_inpatient_total_cost DESC LIMIT {max(50, int(top_n) * 5)}"
    )
    for r in rows:
        score = (
            (int(r.get("ed_visits") or 0)) * 2
            + int(r.get("chronic_condition_count") or 0)
            + (3 if not r.get("has_active_careplan") else 0)
            + (2 if float(r.get("total_outstanding") or 0) > 1000 else 0)
        )
        r["barrier_score"] = score
    rows.sort(key=lambda r: r["barrier_score"], reverse=True)
    return rows[: int(top_n)]


# ─────────────────────────────────────────────
# Tool registry — exposed to the LLM agent loop
# ─────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "pull_demographics": pull_demographics,
    "pull_clinical": pull_clinical,
    "pull_sdoh": pull_sdoh,
    "pull_debt": pull_debt,
    "pull_encounters": pull_encounters,
    "pull_care_gaps": pull_care_gaps,
}

_PLAN_ACTION_ITEM = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "Concrete action."},
        "rationale_barrier": {
            "type": "string",
            "description": "Barrier category from the input that this addresses.",
        },
        "evidence": {
            "type": "string",
            "description": "Quoted fact from the data that justifies this action.",
        },
        "recommended_resources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "link": {"type": "string"}
                },
                "required": ["name", "link"]
            },
            "description": "List of recommended help resources from DUMMY_RESOURCES applicable to this action."
        },
        "owner": {"type": "string"},
        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["action", "rationale_barrier", "evidence", "owner", "priority", "recommended_resources"],
}

SUBMIT_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_care_plan",
        "description": (
            "Submit the final barrier-informed care plan. Call this exactly once "
            "when you have all the information you need. Every action you "
            "include MUST cite a specific barrier and evidence from the input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_summary": {"type": "string"},
                "clinical_actions": {"type": "array", "items": _PLAN_ACTION_ITEM},
                "social_actions": {"type": "array", "items": _PLAN_ACTION_ITEM},
                "financial_actions": {"type": "array", "items": _PLAN_ACTION_ITEM},
                "coordinator_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "3-5 questions only a human with local knowledge can "
                        "answer. NO yes/no, NO data lookups."
                    ),
                },
            },
            "required": [
                "patient_summary",
                "clinical_actions",
                "social_actions",
                "financial_actions",
                "coordinator_questions",
            ],
        },
    },
}

SUBMIT_REVISED_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_revised_care_plan",
        "description": (
            "Submit the revised care plan after incorporating the coordinator's "
            "latest message. Always include a natural-language coordinator_response "
            "so the chat reads like a real conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_summary": {"type": "string"},
                "clinical_actions": {"type": "array", "items": _PLAN_ACTION_ITEM},
                "social_actions": {"type": "array", "items": _PLAN_ACTION_ITEM},
                "financial_actions": {"type": "array", "items": _PLAN_ACTION_ITEM},
                "coordinator_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Fresh questions the agent has for the coordinator after "
                        "this revision. Only include ones that aren't already "
                        "answered by the conversation so far."
                    ),
                },
                "personalization_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Bullet list of what changed in this revision and why, "
                        "tied to the coordinator's most recent feedback."
                    ),
                },
                "coordinator_response": {
                    "type": "string",
                    "description": (
                        "Natural-language reply (1-3 sentences) to the "
                        "coordinator's most recent message — acknowledge what "
                        "you understood, summarize what changed, ask any "
                        "follow-up. This is shown to the coordinator as the "
                        "agent's chat reply."
                    ),
                },
            },
            "required": [
                "patient_summary",
                "clinical_actions",
                "social_actions",
                "financial_actions",
                "personalization_notes",
                "coordinator_response",
            ],
        },
    },
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "pull_demographics",
            "description": (
                "Get the patient's demographic + summary row (age, income, "
                "totals, costs, chronic_condition_count, etc.). Always call this first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pull_clinical",
            "description": (
                "Get active conditions, active medications (with opioid/polypharmacy "
                "flags), and active care plans for the patient."
            ),
            "parameters": {
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pull_sdoh",
            "description": (
                "Get SDOH-related conditions and the most recent PRAPARE survey "
                "responses (housing, transport, food security, work, stress)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pull_debt",
            "description": (
                "Get total outstanding medical debt and unpaid line counts from "
                "claims_transactions for the patient."
            ),
            "parameters": {
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pull_encounters",
            "description": (
                "Get recent emergency / inpatient / urgent-care encounters in a "
                "rolling window (default 12 months) with costs and reasons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 25},
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pull_care_gaps",
            "description": (
                "Detect missing preventive screenings: depression, substance use, "
                "medication reconciliation, annual wellness, alcohol."
            ),
            "parameters": {
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
            },
        },
    },
]
