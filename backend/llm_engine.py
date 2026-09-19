"""LLM recommendation engine for AHEL.

This module contains the real Large Language Model integration used by the
Flask backend. It keeps API keys out of frontend JavaScript and makes the LLM
role explicit for project review/submission.

Supported provider:
- OpenAI-compatible chat completions API through the official openai package.

Environment variables:
- OPENAI_API_KEY: required for live LLM recommendations
- AHEL_LLM_MODEL: optional, defaults to gpt-4.1-mini
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


DEFAULT_MODEL = "gpt-4.1-mini"


def llm_available() -> bool:
    """Return True when the OpenAI SDK and API key are available."""
    return bool(OpenAI is not None and os.getenv("OPENAI_API_KEY"))


def build_prompt(profile: Dict[str, Any], score: Dict[str, Any], gaps: List[Dict[str, Any]]) -> str:
    """Create the structured prompt sent to the LLM."""
    specialization = profile.get("specialization", "Not specified")
    skills = profile.get("skills", [])
    exposure = profile.get("exposure", [])
    first_gap = gaps[0] if gaps else {}

    return f"""
You are generating a personalized AHEL career-readiness recommendation.

Student profile:
- Specialization: {specialization}
- Selected skills: {skills}
- Practical exposure: {exposure}

Scores:
{score}

Highest-priority skill gaps:
{gaps[:5]}

Main gap to explain:
{first_gap}

Write a concise recommendation under 180 words.
Requirements:
1. Mention the specialization.
2. Summarize the readiness level using the supplied score only.
3. Explain the strongest skill gap in natural language.
4. Give exactly three practical next steps.
5. Avoid inventing dataset values or unsupported claims.
""".strip()


def generate_llm_recommendation(profile: Dict[str, Any], score: Dict[str, Any], gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a live LLM recommendation using OpenAI chat completions."""
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to backend/.env or your environment.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("AHEL_LLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    prompt = build_prompt(profile, score, gaps)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an academic AI career-readiness advisor. "
                    "Your output must be specific, evidence-based, concise, and suitable for a student project report."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
    )

    return {
        "recommendation": response.choices[0].message.content.strip(),
        "model_status": "llm_api_used",
        "provider": "OpenAI",
        "model": model,
    }


def validate_openai_key(live_check: bool = True) -> Dict[str, Any]:
    """Validate whether the OpenAI API key is configured and accepted.

    This function is intentionally used by the Flask backend so the UI can show
    a clear error when the key is missing, invalid, expired, or blocked by billing.
    It does not send user data. It only lists available models as a lightweight
    authentication check.
    """
    if OpenAI is None:
        return {
            "configured": False,
            "valid": False,
            "status": "openai_package_missing",
            "message": "The openai Python package is not installed. Run: pip install openai",
            "model": os.getenv("AHEL_LLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_MODEL)),
        }

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("AHEL_LLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_MODEL))

    if not api_key:
        return {
            "configured": False,
            "valid": False,
            "status": "missing_api_key",
            "message": "OPENAI_API_KEY is not configured. Add it to backend/.env.",
            "model": model,
        }

    if not live_check:
        return {
            "configured": True,
            "valid": None,
            "status": "configured_not_checked",
            "message": "OPENAI_API_KEY is configured but has not been live-validated yet.",
            "model": model,
        }

    try:
        client = OpenAI(api_key=api_key, timeout=10.0)
        # Authentication-only check. This confirms the key is accepted without
        # generating text or consuming chat-completion tokens.
        client.models.list()
        return {
            "configured": True,
            "valid": True,
            "status": "valid",
            "message": "OpenAI API key is valid.",
            "model": model,
        }
    except Exception as exc:
        return {
            "configured": True,
            "valid": False,
            "status": "invalid_or_unavailable",
            "message": f"{type(exc).__name__}: {exc}",
            "model": model,
        }


def local_dynamic_recommendation(profile: Dict[str, Any], score: Dict[str, Any], gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback recommendation when no API key is configured.

    This is not an LLM. It is included so the app remains usable offline.
    """
    specialization = profile.get("specialization", "selected specialization")
    skills = profile.get("skills", [])
    total = score.get("total", "N/A")
    first_gap = gaps[0].get("skill") if gaps else "portfolio evidence"

    text = (
        f"Your current AHEL readiness score is {total}/100 for {specialization}. "
        f"The strongest improvement area is {first_gap}. Build one focused project that combines your current skills "
        f"({', '.join(skills[:4]) if skills else 'core technical skills'}) with this missing area. "
        "Then document the project as a short case study, publish the code or evidence, and compare your skills against real job descriptions."
    )

    return {
        "recommendation": text,
        "model_status": "fallback_no_api_key",
        "provider": "Local rule-based fallback",
        "model": "No LLM used",
        "note": "Configure OPENAI_API_KEY to activate live LLM generation.",
    }
