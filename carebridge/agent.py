"""Gemini-powered plan generation + HITL personalization.

Uses Google's `google-genai` SDK with native function calling. Two terminal
tools — `submit_care_plan` and `submit_revised_care_plan` — are how the
model returns its structured output. Research tools (`pull_*`) let the
model fetch additional data when the canned profile leaves something open.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from google.genai import errors as genai_errors
from google.genai import types

from .config import MODEL_CHAIN, gemini_client
from .tools import (
    SUBMIT_PLAN_SCHEMA,
    SUBMIT_REVISED_PLAN_SCHEMA,
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    DUMMY_RESOURCES,
)

PLAN_SYSTEM_PROMPT = """You are a value-based-care coordinator's AI partner. \
Your job is to generate a barrier-informed care plan for a single patient \
that a human coordinator will personalize.

Hard rules:
1. Every recommended action must reference at least one specific barrier \
or care gap from the structured input. NEVER produce generic advice. \
Cite the evidence verbatim ("$111K outstanding debt across 362 unpaid \
claim lines").
2. Group actions into THREE buckets:
   - clinical_actions: what a clinician should do (medication review, \
screening, referral)
   - social_actions: what the coordinator does (housing, transport, food, \
navigation)
   - financial_actions: debt resolution, billing, charity care, payer \
questions
3. Generate 3 to 5 coordinator_questions. These MUST be questions only a \
human with local knowledge of THIS patient or community can answer. \
Forbidden: yes/no confirmations, generic preference questions, anything \
already in the data.
4. Always recommend concrete help resources using the provided list of \
helper resources matching the appropriate barriers. Link the appropriate \
resources in `recommended_resources` for each action.

You have TWO classes of tools:
- pull_* tools — read-only data lookups. Call these only if the structured \
profile is missing something specific you need.
- submit_care_plan — call this EXACTLY ONCE when your plan is ready. This is \
how you return your final answer.

Do not respond in plain text. Every turn must be a tool call."""

REVISE_SYSTEM_PROMPT = """You are in an ongoing chat with a care coordinator. \
They are reviewing the current care plan and giving you feedback in real time \
— flagging actions that won't work, sharing local knowledge you can't see in \
the data, asking clarifying questions, suggesting changes.

Each turn:
1. Read the current plan, the conversation so far, and the coordinator's NEW \
message.
2. If they're flagging a problem with an action, REWRITE that action — don't \
just append a note. If they're sharing context, fold it into the relevant \
action's evidence. If they're asking a question, answer it concisely in \
coordinator_response and only update the plan if needed.
3. If you need patient data to answer their question, call the relevant \
pull_* tool first.
4. Always finish by calling submit_revised_care_plan with:
   - the full updated plan (not just the diff), ensure actions still include `recommended_resources` matching the barriers if appropriate.
   - personalization_notes: bullet list of what changed *this turn*
   - coordinator_response: your natural-language reply to the coordinator's \
last message (acknowledge, summarize the change, ask follow-up if useful)

Do not respond in plain text. Every turn must end with submit_revised_care_plan."""


def _to_gemini_declaration(openai_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style tool schema into a Gemini function declaration."""
    fn = openai_schema["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _build_tools(extra_schemas: list[dict[str, Any]]) -> list[types.Tool]:
    declarations = [
        _to_gemini_declaration(s) for s in (TOOL_SCHEMAS + extra_schemas)
    ]
    return [types.Tool(function_declarations=declarations)]


def _exec_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _agent_loop(
    *,
    system_prompt: str,
    user_text: str,
    extra_schemas: list[dict[str, Any]],
    submit_tool_name: str,
    max_iters: int = 8,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Drive a Gemini tool-calling loop until the model calls the submit tool.

    Returns the parsed arguments of that submit call.
    """
    client = gemini_client()
    tools = _build_tools(extra_schemas)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
        temperature=0.2,
        max_output_tokens=4096,
    )

    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_text)])
    ]

    # Track which models in the chain are exhausted for the rest of this call,
    # so we don't keep retrying a dead one across loop iterations.
    exhausted_models: set[str] = set()

    def _generate_with_retry() -> Any:
        last_exc: Exception | None = None
        for model_name in MODEL_CHAIN:
            if model_name in exhausted_models:
                continue
            for delay in (0, 3, 10):
                if delay:
                    time.sleep(delay)
                try:
                    return client.models.generate_content(
                        model=model_name,
                        contents=history,
                        config=config,
                    )
                except genai_errors.ServerError as exc:
                    last_exc = exc
                    if log:
                        log(f"  [retry] {model_name} 503 — backing off {delay or 3}s")
                    continue
                except genai_errors.ClientError as exc:
                    last_exc = exc
                    if exc.code == 429:
                        if log:
                            log(
                                f"  [fallback] {model_name} quota exhausted, "
                                "trying next model in chain"
                            )
                        exhausted_models.add(model_name)
                        break
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry loop produced no response")

    for _ in range(max_iters):
        resp = _generate_with_retry()

        candidate = resp.candidates[0] if resp.candidates else None
        parts = list((candidate.content.parts if candidate and candidate.content else []) or [])

        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            # Push back: insist on a tool call.
            history.append(
                types.Content(role="model", parts=parts or [types.Part(text="")])
            )
            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "You must respond with a function call, not "
                                f"plain text. If the plan is ready, call "
                                f"{submit_tool_name}."
                            )
                        )
                    ],
                )
            )
            continue

        # Persist the model's full turn (parts containing function_call entries).
        history.append(types.Content(role="model", parts=parts))

        # Did the model call the terminal submit tool? Capture and stop.
        for fc in function_calls:
            if fc.name == submit_tool_name:
                if log:
                    log(f"  [tool] {fc.name}(...)  ← submit, terminating")
                return dict(fc.args or {})

        # Otherwise execute every research call, append responses, loop.
        response_parts: list[types.Part] = []
        for fc in function_calls:
            args = dict(fc.args or {})
            if log:
                log(f"  [tool] {fc.name}({json.dumps(args)})")
            output = _exec_tool(fc.name, args)
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": output},
                )
            )
        history.append(types.Content(role="user", parts=response_parts))

    raise RuntimeError(
        f"Agent loop exhausted {max_iters} iterations without "
        f"calling {submit_tool_name}"
    )


def generate_plan(
    profile: dict[str, Any],
    barriers: list[dict[str, Any]],
    care_gaps: list[dict[str, Any]],
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """First-pass plan via Gemini using real tool calls."""
    user_payload = {
        "patient": {
            "id": profile["demographics"].get("id"),
            "name": f"{profile['demographics'].get('first')} "
            f"{profile['demographics'].get('last')}",
            "age": profile["demographics"].get("age"),
            "income": profile["demographics"].get("income"),
            "city": profile["demographics"].get("city"),
            "zip": profile["demographics"].get("zip"),
            "ed_visits": profile["demographics"].get("ed_visits"),
            "inpatient_visits": profile["demographics"].get(
                "inpatient_visits"
            ),
            "chronic_condition_count": profile["demographics"].get(
                "chronic_condition_count"
            ),
        },
        "barriers": barriers,
        "care_gaps": care_gaps,
        "active_conditions": [
            c.get("DESCRIPTION")
            for c in profile["clinical"].get("active_conditions", [])
        ],
        "active_medications": [
            c.get("DESCRIPTION")
            for c in profile["clinical"].get("active_medications", [])
        ],
        "has_active_careplan": profile["clinical"].get("has_active_careplan"),
        "prapare": profile["sdoh"].get("prapare", {}),
        "debt": profile.get("debt", {}),
        "available_resources": DUMMY_RESOURCES,
    }

    user_text = (
        "Generate the barrier-informed care plan for this patient.\n\n"
        "Here is the structured profile:\n"
        f"{json.dumps(user_payload, indent=2, default=str)}\n\n"
        "When the plan is complete, call submit_care_plan."
    )

    return _agent_loop(
        system_prompt=PLAN_SYSTEM_PROMPT,
        user_text=user_text,
        extra_schemas=[SUBMIT_PLAN_SCHEMA],
        submit_tool_name="submit_care_plan",
        log=log,
    )


def revise_plan(
    *,
    profile: dict[str, Any],
    barriers: list[dict[str, Any]],
    care_gaps: list[dict[str, Any]],
    current_plan: dict[str, Any],
    conversation: list[dict[str, str]],
    new_message: str,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one chat turn — coordinator says something, agent revises the plan.

    `conversation` is the chat so far (excluding the new message): a list of
    {"coordinator": "...", "agent_response": "...", "personalization_notes": [...]}
    entries, oldest first. The new message is appended in the prompt.
    """
    history_lines: list[str] = []
    for i, turn in enumerate(conversation, start=1):
        history_lines.append(f"[Turn {i}] Coordinator: {turn.get('coordinator', '')}")
        if turn.get("agent_response"):
            history_lines.append(f"          Agent: {turn['agent_response']}")
        for note in turn.get("personalization_notes") or []:
            history_lines.append(f"          ↳ change: {note}")

    payload = {
        "patient": {
            "id": profile["demographics"].get("id"),
            "name": f"{profile['demographics'].get('first')} "
            f"{profile['demographics'].get('last')}",
            "city": profile["demographics"].get("city"),
            "zip": profile["demographics"].get("zip"),
        },
        "barriers": barriers,
        "care_gaps": care_gaps,
        "current_plan": current_plan,
        "conversation_so_far": history_lines or ["(none yet)"],
        "new_coordinator_message": new_message,
        "available_resources": DUMMY_RESOURCES,
    }

    user_text = (
        "The coordinator just sent a new message in the ongoing chat. "
        "Review the current plan, the conversation so far, and the new "
        "message, then revise.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Use pull_* tools if you need patient data not already shown. "
        "Then call submit_revised_care_plan with the full updated plan, "
        "personalization_notes describing what changed THIS TURN, and a "
        "natural-language coordinator_response."
    )

    return _agent_loop(
        system_prompt=REVISE_SYSTEM_PROMPT,
        user_text=user_text,
        extra_schemas=[SUBMIT_REVISED_PLAN_SCHEMA],
        submit_tool_name="submit_revised_care_plan",
        log=log,
    )
