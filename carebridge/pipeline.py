"""End-to-end orchestration of the 5 stages.

Stage 1 RESOLVE → Stage 2 PROFILE → Stage 3 ANALYZE → Stage 4 GENERATE
→ Stage 5 PERSONALIZE.

Each stage is a pure function so a future UI can import them piecemeal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .agent import generate_plan, revise_plan
from .barriers import identify_barriers, identify_care_gaps
from .db import resolve_patient
from .tools import (
    pull_care_gaps,
    pull_clinical,
    pull_debt,
    pull_demographics,
    pull_encounters,
    pull_sdoh,
)

OUTPUT_DIR = Path(os.environ.get("CAREBRIDGE_OUTPUT", "output"))


def banner(label: str, log: Callable[[str], None]) -> None:
    log("")
    log("─" * 70)
    log(f"  {label}")
    log("─" * 70)


def build_profile(patient_id: str) -> dict[str, Any]:
    """Stage 2: pull the full picture in one shot."""
    return {
        "demographics": pull_demographics(patient_id),
        "clinical": pull_clinical(patient_id),
        "sdoh": pull_sdoh(patient_id),
        "debt": pull_debt(patient_id),
        "encounters": pull_encounters(patient_id),
        "care_gaps": pull_care_gaps(patient_id),
    }


def collect_coordinator_answers(
    questions: list[str],
    *,
    interactive: bool = True,
    answers: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, str]]:
    """Prompt the coordinator interactively, or accept pre-supplied answers."""
    out: list[dict[str, str]] = []
    log = log or (lambda _msg: None)
    answers = answers or {}
    for i, q in enumerate(questions, start=1):
        if interactive:
            log("")
            log(f"Coordinator question {i}/{len(questions)}:")
            log(f"  {q}")
            try:
                a = input("  Your answer (blank to skip): ").strip()
            except EOFError:
                a = ""
        else:
            a = answers.get(q) or answers.get(str(i)) or ""
        out.append({"question": q, "answer": a or "(no answer provided)"})
    return out


def run(
    patient_input: str,
    *,
    interactive_hitl: bool = True,
    preset_answers: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for one patient. Returns the final plan dict."""
    log = log or print

    banner("Stage 1 — RESOLVE patient", log)
    patient_row = resolve_patient(patient_input)
    pid = patient_row["id"]
    log(
        f"  → matched: {patient_row.get('first')} {patient_row.get('last')} "
        f"({pid})"
    )
    log(
        f"  → ed_visits={patient_row.get('ed_visits')} "
        f"chronic={patient_row.get('chronic_condition_count')} "
        f"active_careplan={patient_row.get('has_active_careplan')}"
    )

    banner("Stage 2 — PROFILE (clinical + social + financial)", log)
    profile = build_profile(pid)
    c = profile["clinical"]
    s = profile["sdoh"]
    d = profile["debt"]
    log(
        f"  → {len(c['active_conditions'])} active conditions · "
        f"{c['active_med_count']} active meds "
        f"({c['active_opioid_count']} opioids)"
    )
    log(
        f"  → SDOH conditions: {len(s['sdoh_conditions'])} · "
        f"PRAPARE answers: {len(s['prapare'])}"
    )
    log(
        f"  → outstanding debt: ${d['total_outstanding']:,.0f} "
        f"across {d['unpaid_lines']} lines"
    )

    banner("Stage 3 — ANALYZE barriers + care gaps", log)
    barriers = [b.to_dict() for b in identify_barriers(profile)]
    care_gaps = identify_care_gaps(profile)
    log(f"  → identified {len(barriers)} barriers:")
    for b in barriers:
        log(f"    · [{b['severity']:>6}] {b['category']}: {b['summary']}")
    log(f"  → {len(care_gaps)} care gaps:")
    for g in care_gaps:
        log(f"    · [{g['priority']:>6}] {g['screening']}")

    banner("Stage 4 — GENERATE barrier-informed care plan (Gemini)", log)
    plan = generate_plan(profile, barriers, care_gaps, log=log)
    _print_plan(plan, log)

    banner("Stage 5 — PERSONALIZE with coordinator local knowledge", log)
    questions = plan.get("coordinator_questions", []) or []
    conversation: list[dict[str, Any]] = []
    final_plan = plan
    if not questions:
        log("  (model returned no coordinator_questions — skipping HITL)")
        answers: list[dict[str, str]] = []
    else:
        answers = collect_coordinator_answers(
            questions,
            interactive=interactive_hitl,
            answers=preset_answers,
            log=log,
        )
        # Turn structured Q&A into a single chat message for revise_plan.
        message_lines = []
        for a in answers:
            message_lines.append(f"Q: {a['question']}")
            message_lines.append(f"A: {a['answer']}")
        new_message = "\n".join(message_lines) or "(no answers provided)"
        log("")
        log("  Re-prompting model with coordinator answers...")
        final_plan = revise_plan(
            profile=profile,
            barriers=barriers,
            care_gaps=care_gaps,
            current_plan=plan,
            conversation=conversation,
            new_message=new_message,
            log=log,
        )
        conversation.append(
            {
                "coordinator": new_message,
                "agent_response": final_plan.get("coordinator_response", ""),
                "personalization_notes": final_plan.get(
                    "personalization_notes", []
                ),
            }
        )
        log("")
        log("  Revised plan:")
        _print_plan(final_plan, log)
        if final_plan.get("coordinator_response"):
            log(f"  Agent reply: {final_plan['coordinator_response']}")
        if final_plan.get("personalization_notes"):
            log("  Personalization notes:")
            for n in final_plan["personalization_notes"]:
                log(f"    · {n}")

    artifact = {
        "patient_id": pid,
        "patient_name": f"{patient_row.get('first')} {patient_row.get('last')}",
        "profile_summary": {
            "active_condition_count": len(c["active_conditions"]),
            "active_med_count": c["active_med_count"],
            "sdoh_condition_count": len(s["sdoh_conditions"]),
            "outstanding_debt": d["total_outstanding"],
        },
        "barriers": barriers,
        "care_gaps": care_gaps,
        "initial_plan": plan,
        "coordinator_answers": answers,
        "conversation": conversation,
        "final_plan": final_plan,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pid}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    log("")
    log(f"Saved artifact → {out_path}")

    return artifact


def _print_plan(plan: dict[str, Any], log: Callable[[str], None]) -> None:
    if not isinstance(plan, dict):
        log(f"  (unexpected plan type: {type(plan).__name__})")
        return
    if plan.get("patient_summary"):
        log(f"  summary: {plan['patient_summary']}")
    for bucket in ("clinical_actions", "social_actions", "financial_actions"):
        items = plan.get(bucket) or []
        if not items:
            continue
        log(f"  {bucket}:")
        for it in items:
            log(
                f"    · [{it.get('priority', '?'):>6}] "
                f"{it.get('action', '?')}"
            )
            if it.get("rationale_barrier"):
                log(f"        ↳ barrier: {it['rationale_barrier']}")
            if it.get("evidence"):
                log(f"        ↳ evidence: {it['evidence']}")
    qs = plan.get("coordinator_questions") or []
    if qs:
        log("  coordinator_questions:")
        for q in qs:
            log(f"    ? {q}")
