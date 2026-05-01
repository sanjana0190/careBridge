"""Central configuration: env loading, Gemini client, D1 endpoint."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

load_dotenv()

D1_API_URL = "https://uic-hackathon-data.christian-7f4.workers.dev/query"

# Fallback chain: walked top-to-bottom on quota/availability errors.
# Override the primary by setting CAREBRIDGE_MODEL.
DEFAULT_MODEL_CHAIN = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
_override = os.environ.get("CAREBRIDGE_MODEL")
MODEL_CHAIN: tuple[str, ...] = (
    (_override, *DEFAULT_MODEL_CHAIN) if _override else DEFAULT_MODEL_CHAIN
)
MODEL = MODEL_CHAIN[0]  # primary, kept for backwards compatibility


def _resolve_gemini_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "GEMINI_API_KEY not found in environment. Set it in .env "
        "(GEMINI_API_KEY=AIza...) or as an env var."
    )


@lru_cache(maxsize=1)
def gemini_client() -> "genai.Client":
    return genai.Client(api_key=_resolve_gemini_key())
