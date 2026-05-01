"""careBridge web UI — minimal Flask frontend with a multi-turn HITL chat.

Pages:
  GET  /           name input + link to patient list
  GET  /patients   table of all patients (copy ID/name from here)
  POST /run        run stages 1-4 for the entered name; show plan + chat box
  POST /chat       coordinator sends a message; agent revises the plan
  POST /done       coordinator finishes; save artifact and show summary
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, abort, redirect, render_template, request, url_for

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from carebridge.agent import generate_plan, revise_plan
from carebridge.barriers import identify_barriers, identify_care_gaps
from carebridge.db import PatientNotFound, query, resolve_patient
from carebridge.pipeline import build_profile

app = Flask(__name__)

DRAFT_DIR = Path("output") / "drafts"
DRAFT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("output")


def _capture_log() -> tuple[list[str], Any]:
    buf: list[str] = []

    def log(msg: str) -> None:
        buf.append(str(msg))

    return buf, log


def _save_draft(draft_id: str, state: dict[str, Any]) -> None:
    path = DRAFT_DIR / f"{draft_id}.json"
    path.write_text(json.dumps(state, default=str, indent=2), encoding="utf-8")


def _load_draft(draft_id: str) -> dict[str, Any]:
    path = DRAFT_DIR / f"{draft_id}.json"
    if not path.exists():
        abort(404, "draft expired or not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _render_session(state: dict[str, Any], *, error: str | None = None) -> Any:
    return render_template(
        "result.html",
        patient=state["patient_row"],
        barriers=state["barriers"],
        care_gaps=state["care_gaps"],
        plan=state["current_plan"],
        conversation=state.get("conversation", []),
        log=state.get("log", []),
        draft_id=state["draft_id"],
        error=error,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/patients")
def patients():
    rows = query(
        "SELECT id, first, last, age, gender, city, ed_visits, "
        "chronic_condition_count, has_active_careplan, "
        "ed_inpatient_total_cost FROM patient_summary "
        "ORDER BY ed_inpatient_total_cost DESC LIMIT 200"
    )
    return render_template("patients.html", rows=rows)


@app.route("/run", methods=["POST"])
def run():
    name = (request.form.get("name") or "").strip()
    if not name:
        return redirect(url_for("index"))

    log_buf, log = _capture_log()

    try:
        patient_row = resolve_patient(name)
    except PatientNotFound as exc:
        return render_template("index.html", error=str(exc), prefill=name)

    pid = patient_row["id"]
    log(f"Resolved → {patient_row.get('first')} {patient_row.get('last')} ({pid})")

    profile = build_profile(pid)
    log(
        f"Profile pulled: {len(profile['clinical']['active_conditions'])} active "
        f"conditions, {profile['clinical']['active_med_count']} meds, "
        f"{len(profile['sdoh']['sdoh_conditions'])} SDOH conditions, "
        f"${profile['debt']['total_outstanding']:,.0f} outstanding debt"
    )

    barriers = [b.to_dict() for b in identify_barriers(profile)]
    care_gaps = identify_care_gaps(profile)

    log("Calling Gemini agent for initial plan…")
    try:
        plan = generate_plan(profile, barriers, care_gaps, log=log)
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "index.html",
            error=f"Plan generation failed: {exc}",
            prefill=name,
        )

    draft_id = uuid.uuid4().hex
    state = {
        "draft_id": draft_id,
        "patient_id": pid,
        "patient_row": patient_row,
        "profile": profile,
        "barriers": barriers,
        "care_gaps": care_gaps,
        "initial_plan": plan,
        "current_plan": plan,
        "conversation": [],
        "log": log_buf,
    }
    _save_draft(draft_id, state)

    return _render_session(state)


@app.route("/chat", methods=["POST"])
def chat():
    draft_id = request.form.get("draft_id")
    message = (request.form.get("message") or "").strip()
    if not draft_id:
        abort(400, "missing draft_id")

    state = _load_draft(draft_id)

    if not message:
        return _render_session(state, error="Type a message to send to the agent.")

    log_buf, log = _capture_log()
    log_buf.extend(state.get("log", []))
    log("")
    log(f"Coordinator: {message}")

    try:
        revised = revise_plan(
            profile=state["profile"],
            barriers=state["barriers"],
            care_gaps=state["care_gaps"],
            current_plan=state["current_plan"],
            conversation=state["conversation"],
            new_message=message,
            log=log,
        )
    except Exception as exc:  # noqa: BLE001
        state["log"] = log_buf + [f"ERROR: {exc}"]
        _save_draft(draft_id, state)
        return _render_session(state, error=f"Revision failed: {exc}")

    state["conversation"].append(
        {
            "coordinator": message,
            "agent_response": revised.get("coordinator_response") or "",
            "personalization_notes": revised.get("personalization_notes") or [],
        }
    )
    state["current_plan"] = revised
    state["log"] = log_buf
    _save_draft(draft_id, state)

    return _render_session(state)


@app.route("/done", methods=["POST"])
def done():
    draft_id = request.form.get("draft_id")
    if not draft_id:
        abort(400, "missing draft_id")
    state = _load_draft(draft_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{state['patient_id']}.json"
    out_path.write_text(
        json.dumps(state, default=str, indent=2), encoding="utf-8"
    )
    (DRAFT_DIR / f"{draft_id}.json").unlink(missing_ok=True)

    return render_template(
        "final.html",
        patient=state["patient_row"],
        barriers=state["barriers"],
        care_gaps=state["care_gaps"],
        initial_plan=state["initial_plan"],
        final_plan=state["current_plan"],
        conversation=state["conversation"],
        log=state["log"],
        artifact_path=str(out_path),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
