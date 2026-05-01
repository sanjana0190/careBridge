"""
UIC Healthcare Hackathon — Agent Example (Python)
==================================================
Pattern: System Prompt + Tool Definition + Agentic Loop

NO API KEY? Use the Cloudflare Workers path instead — it's free:
  See docs/cloudflare_deploy.md — Workers AI runs on Cloudflare's free tier,
  no external API key needed.

If you want to run locally, you need a key from one of:
  Free  — Groq:      console.groq.com      (OpenAI-compatible, generous free tier)
  Paid  — Anthropic: console.anthropic.com (this example uses Anthropic SDK)
  Paid  — OpenAI:    platform.openai.com

Setup (Anthropic):
  pip install anthropic requests
  export ANTHROPIC_API_KEY=your_key_here
  python agent_example.py
"""

import os
import json
import requests
import anthropic

# ─────────────────────────────────────────────
# CONFIGURATION — update the Worker URL once it's deployed
# ─────────────────────────────────────────────

D1_API_URL = "https://uic-hackathon-data.christian-7f4.workers.dev/query"


# ─────────────────────────────────────────────
# STEP 1: TOOLS — what can your agent actually do?
#
# Each tool is a Python function (what runs) + a schema (what the LLM sees).
# Add more tools here as you build out your agent.
# ─────────────────────────────────────────────

def query_database(sql: str) -> dict:
    """Execute a read-only SQL query against the hackathon D1 database."""
    try:
        response = requests.post(D1_API_URL, json={"sql": sql}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


# Tool schemas — the LLM reads these to know what it can call and how
TOOLS = [
    {
        "name": "query_database",
        "description": (
            "Execute a SQL SELECT query against the patient dataset. "
            "Returns JSON with a 'results' array. "
            "Available tables: patients, encounters, conditions, medications, "
            "observations, procedures, claims_transactions, careplans. "
            "Use the 'patient_summary' view as your starting point — it has "
            "one row per patient with pre-computed visit counts, costs, and flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQL SELECT statement. Always include a LIMIT clause.",
                }
            },
            "required": ["sql"],
        },
    },
    # ── ADD YOUR OWN TOOLS HERE ──────────────────────────────────────────────
    # Example: a tool that formats a care plan for a coordinator to review,
    # a tool that checks an external resource directory, a tool that sends
    # an alert. Any tool without an "execute" function becomes a human-in-the-
    # loop confirmation step in frameworks that support it.
]


# ─────────────────────────────────────────────
# STEP 2: SYSTEM PROMPT — the agent's job description
#
# This is the most important thing you'll write.
# Tell it: who it is, what it's trying to achieve, how to use the tools,
# and when to stop and ask a human.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a healthcare data analyst assistant helping care coordinators \
at a value-based primary care practice.

Your goal is to help coordinators find patients who need intervention, understand \
what's driving high costs, and surface barriers keeping patients from accessing care.

You have access to a database of 117 synthetic patients. Key facts:
- Total costs: $27.9M across 8,316 encounters
- Inpatient visits = 35% of cost from only 2.8% of encounters
- 83% of patients have at least 1 ED visit
- Social factors (housing, food, transport, employment) are in conditions and observations

How to investigate:
1. Start with patient_summary — sort by ed_inpatient_total_cost or total_visits
2. Use the PATIENT column as the join key across most tables
   (exception: claims_transactions uses PATIENTID)
3. Look for both clinical signals (conditions, meds) AND social signals (SDOH)
4. Always show your reasoning before drawing conclusions

Human-in-the-loop rule: Before recommending any action (outreach, care plan change, \
escalation), summarize your findings and ask the coordinator to confirm before proceeding.
"""


# ─────────────────────────────────────────────
# STEP 3: THE AGENT LOOP
#
# The loop:
#   1. Send messages to the LLM
#   2. If it wants to use a tool → run the tool, add results, loop again
#   3. If it's done (end_turn) → print the response and exit
# ─────────────────────────────────────────────

def run_agent(user_message: str) -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": user_message}]

    print(f"\nUser: {user_message}\n{'─' * 60}")

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append the assistant's response to the conversation history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Agent is done — print its final answer
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent: {block.text}")
            break

        elif response.stop_reason == "tool_use":
            # Agent called one or more tools — execute them and return results
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n[Tool call: {block.name}]")
                    print(f"  Input: {json.dumps(block.input, indent=2)}")

                    # Route the tool call to the right function
                    if block.name == "query_database":
                        result = query_database(block.input["sql"])
                    else:
                        result = {"error": f"Unknown tool: {block.name}"}

                    print(f"  Result: {len(result.get('results', []))} rows returned")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            # Feed tool results back into the conversation
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            print(f"Stopped: {response.stop_reason}")
            break


# ─────────────────────────────────────────────
# Try these starting prompts — pick the one that matches your challenge prompt
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Prompt 1: Preventable Visit Detector
    run_agent(
        "Find the patients most at risk of a preventable ED visit. "
        "Consider both clinical factors (chronic conditions, no care plan) "
        "and social factors (SDOH, financial barriers). "
        "Rank the top 5 and explain why each one is high risk."
    )

    # Prompt 2: Cost Explainer
    # run_agent(
    #     "I need to understand why Giovanni Paucek's care is so expensive. "
    #     "Walk me through the breakdown — what types of encounters, "
    #     "what conditions, and which costs look reducible."
    # )

    # Prompt 3: Care Barrier Agent
    # run_agent(
    #     "Pull a full profile for Lindsay Brekke. "
    #     "What clinical and social barriers are preventing her from getting consistent care? "
    #     "Generate a barrier-informed care plan for me to review."
    # )