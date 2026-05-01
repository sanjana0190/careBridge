"""Rule-based barrier identification with evidence pointers.

Each Barrier carries category, severity, a one-line summary, and `evidence`
that quotes the source rows that triggered the flag — so the downstream
care plan can cite specific facts instead of generic boilerplate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .tools import OPIOID_KEYWORDS


@dataclass
class Barrier:
    category: str
    severity: str  # "low" | "medium" | "high"
    summary: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prapare_value(prapare: dict[str, dict[str, Any]], key: str) -> str | None:
    entry = prapare.get(key)
    return (entry or {}).get("value") if entry else None


def identify_barriers(profile: dict[str, Any]) -> list[Barrier]:
    """Run all rules against a fully-pulled profile dict."""
    demo = profile.get("demographics", {})
    clinical = profile.get("clinical", {})
    sdoh = profile.get("sdoh", {})
    debt = profile.get("debt", {})
    encounters = profile.get("encounters", {})

    sdoh_descs = [
        (c.get("DESCRIPTION") or "") for c in sdoh.get("sdoh_conditions", [])
    ]
    prapare = sdoh.get("prapare", {})

    barriers: list[Barrier] = []

    # Financial
    outstanding = float(debt.get("total_outstanding") or 0)
    if outstanding > 0:
        if outstanding > 5000:
            sev = "high"
        elif outstanding > 1000:
            sev = "medium"
        else:
            sev = "low"
        barriers.append(
            Barrier(
                category="financial_debt",
                severity=sev,
                summary=f"${outstanding:,.0f} in outstanding medical debt across "
                f"{debt.get('unpaid_lines', 0)} unpaid claim lines.",
                evidence=[
                    f"oldest unpaid: {debt.get('oldest_unpaid_date')}",
                    f"newest unpaid: {debt.get('newest_unpaid_date')}",
                ],
            )
        )

    # Housing instability
    housing_evidence: list[str] = []
    for d in sdoh_descs:
        dl = d.lower()
        if "housing" in dl or "homeless" in dl:
            housing_evidence.append(f"condition: {d}")
    worried = _prapare_value(
        prapare, "Are you worried about losing your housing?"
    )
    if worried and "yes" in str(worried).lower():
        housing_evidence.append(f"PRAPARE: worried about losing housing → {worried}")
    housing_status = _prapare_value(prapare, "Housing status")
    if housing_status and any(
        s in str(housing_status).lower() for s in ("homeless", "someone else")
    ):
        housing_evidence.append(f"PRAPARE housing status: {housing_status}")
    if housing_evidence:
        barriers.append(
            Barrier(
                category="housing_instability",
                severity="high",
                summary="Patient shows housing-related risk factors.",
                evidence=housing_evidence,
            )
        )

    # Transportation
    transport_evidence: list[str] = []
    for d in sdoh_descs:
        if "transport" in d.lower():
            transport_evidence.append(f"condition: {d}")
    car = _prapare_value(
        prapare, "Do you have a car or can you get a ride in a car?"
    )
    if car and "no" in str(car).lower():
        transport_evidence.append(f"PRAPARE: car/ride access → {car}")
    if transport_evidence:
        barriers.append(
            Barrier(
                category="transportation",
                severity="high",
                summary="Patient lacks reliable transportation to appointments.",
                evidence=transport_evidence,
            )
        )

    # Food insecurity
    food = _prapare_value(
        prapare,
        "How hard is it for you to pay for the very basics like food, "
        "housing, medical care, and heating?",
    )
    food_evidence: list[str] = []
    if food and any(s in str(food).lower() for s in ("hard", "very hard", "somewhat")):
        food_evidence.append(f"PRAPARE basics-affordability: {food}")
    for d in sdoh_descs:
        if "food" in d.lower():
            food_evidence.append(f"condition: {d}")
    if food_evidence:
        barriers.append(
            Barrier(
                category="food_insecurity",
                severity="high",
                summary="Patient has trouble affording food / basic needs.",
                evidence=food_evidence,
            )
        )

    # Employment
    work = _prapare_value(prapare, "What is your current work situation?")
    employment_evidence: list[str] = []
    if work and "unemploy" in str(work).lower():
        employment_evidence.append(f"PRAPARE work: {work}")
    for d in sdoh_descs:
        dl = d.lower()
        if "unemploy" in dl or "not in labor force" in dl:
            employment_evidence.append(f"condition: {d}")
    if employment_evidence:
        barriers.append(
            Barrier(
                category="employment",
                severity="medium",
                summary="Patient is unemployed / out of labor force.",
                evidence=employment_evidence,
            )
        )

    # Social isolation / stress
    isolation_evidence: list[str] = []
    for d in sdoh_descs:
        dl = d.lower()
        if "social contact" in dl or "stress" in dl:
            isolation_evidence.append(f"condition: {d}")
    stress = _prapare_value(prapare, "Stress level")
    if stress and any(
        s in str(stress).lower() for s in ("quite a bit", "very much", "somewhat")
    ):
        isolation_evidence.append(f"PRAPARE stress level: {stress}")
    if isolation_evidence:
        barriers.append(
            Barrier(
                category="social_isolation_stress",
                severity="medium",
                summary="Elevated stress / social-isolation indicators.",
                evidence=isolation_evidence,
            )
        )

    # Polypharmacy
    med_count = int(clinical.get("active_med_count") or 0)
    if med_count >= 5:
        names = [
            (m.get("DESCRIPTION") or "")[:80]
            for m in clinical.get("active_medications", [])[:8]
        ]
        barriers.append(
            Barrier(
                category="polypharmacy",
                severity="medium",
                summary=f"Polypharmacy: {med_count} active medications "
                "increases adherence + interaction risk.",
                evidence=[f"active med: {n}" for n in names],
            )
        )

    # Opioid exposure
    if int(clinical.get("active_opioid_count") or 0) > 0:
        opioids = [
            (m.get("DESCRIPTION") or "")
            for m in clinical.get("active_medications", [])
            if any(
                k in (m.get("DESCRIPTION") or "").lower() for k in OPIOID_KEYWORDS
            )
        ]
        barriers.append(
            Barrier(
                category="opioid_exposure",
                severity="high",
                summary="Active opioid prescription on record.",
                evidence=[f"opioid: {n}" for n in opioids[:5]],
            )
        )

    # Care plan absence with chronic conditions
    chronic = int(demo.get("chronic_condition_count") or 0)
    if not clinical.get("has_active_careplan") and chronic >= 3:
        barriers.append(
            Barrier(
                category="no_active_care_plan",
                severity="high",
                summary=f"{chronic} chronic conditions but NO active care plan.",
                evidence=[f"chronic_condition_count = {chronic}"],
            )
        )

    # High utilizer (lifetime ED count from summary, since data is historical)
    lifetime_ed = int(demo.get("ed_visits") or 0)
    if lifetime_ed >= 5:
        sev = "high" if lifetime_ed >= 15 else "medium"
        barriers.append(
            Barrier(
                category="high_utilizer",
                severity=sev,
                summary=f"{lifetime_ed} lifetime ED visits — pattern of "
                "acute-care reliance rather than primary-care follow-through.",
                evidence=[
                    f"ed_visits = {lifetime_ed}",
                    f"inpatient_visits = {demo.get('inpatient_visits')}",
                    f"ed_inpatient_total_cost = ${float(demo.get('ed_inpatient_total_cost') or 0):,.0f}",
                ],
            )
        )

    # Low income
    income = demo.get("income")
    try:
        income_v = float(income) if income is not None else None
    except (TypeError, ValueError):
        income_v = None
    if income_v is not None and income_v < 25000:
        barriers.append(
            Barrier(
                category="low_income",
                severity="medium",
                summary=f"Annual income ${income_v:,.0f} is below the federal "
                "poverty line for most household sizes.",
                evidence=[f"patients.INCOME = {income_v}"],
            )
        )

    return barriers


def identify_care_gaps(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the screening dict into an ordered gap list."""
    screenings = (profile.get("care_gaps") or {}).get("screenings", {})
    gaps = []
    priority = {
        "depression": "high",
        "substance_use": "high",
        "medication_reconciliation": "medium",
        "annual_wellness": "medium",
        "alcohol": "medium",
    }
    for key, info in screenings.items():
        if not info.get("present"):
            gaps.append(
                {
                    "screening": key,
                    "priority": priority.get(key, "low"),
                    "summary": f"No record of {key.replace('_', ' ')} on file.",
                }
            )
    return gaps
