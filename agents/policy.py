"""Policy presets and aggregate-only natural-language threshold interpretation.

Gemini may translate a user's policy request into bounded numeric thresholds.
The release decision remains deterministic code in ``orchestrator.decide``.
"""

from __future__ import annotations

import re
from typing import Dict

from pydantic import BaseModel, Field

import llm
from config import get_settings
from models import Thresholds


class PolicyInterpretation(BaseModel):
    thresholds: Thresholds
    policy_name: str = "Custom policy"
    explanation: str = ""
    source: str = "deterministic"


class _LLMPolicy(BaseModel):
    fidelity_min: float = Field(ge=0.0, le=1.0)
    tstr_min: float = Field(ge=0.0, le=1.0)
    query_equivalence_min: float = Field(ge=0.0, le=1.0)
    reid_max: float = Field(ge=0.0, le=1.0)
    membership_max: float = Field(ge=0.0, le=1.0)
    attribute_inference_max: float = Field(ge=0.0, le=1.0)
    linkage_max: float = Field(ge=0.0, le=1.0)
    near_duplicate_max: float = Field(ge=0.0, le=1.0)
    max_rounds: int = Field(ge=1, le=5)
    policy_name: str
    explanation: str


PRESETS: Dict[str, dict] = {
    "balanced": {
        "name": "Balanced",
        "description": "The default balance of analytical usefulness and attack resistance.",
        "thresholds": Thresholds(),
    },
    "privacy_first": {
        "name": "Privacy first",
        "description": "Tighter privacy ceilings with modestly relaxed utility floors.",
        "thresholds": Thresholds(
            fidelity_min=0.80,
            tstr_min=0.82,
            query_equivalence_min=0.92,
            reid_max=0.005,
            membership_max=0.55,
            attribute_inference_max=0.15,
            linkage_max=0.08,
            near_duplicate_max=0.05,
        ),
    },
    "analytics_first": {
        "name": "Analytics first",
        "description": "Higher utility floors while retaining explicit privacy release ceilings.",
        "thresholds": Thresholds(
            fidelity_min=0.90,
            tstr_min=0.90,
            query_equivalence_min=0.97,
            reid_max=0.015,
            membership_max=0.62,
            attribute_inference_max=0.28,
            linkage_max=0.18,
            near_duplicate_max=0.12,
        ),
    },
}


def presets_payload() -> list[dict]:
    return [
        {
            "id": preset_id,
            "name": item["name"],
            "description": item["description"],
            "thresholds": item["thresholds"].model_dump(),
        }
        for preset_id, item in PRESETS.items()
    ]


def preset(preset_id: str) -> PolicyInterpretation:
    item = PRESETS.get(preset_id, PRESETS["balanced"])
    return PolicyInterpretation(
        thresholds=item["thresholds"],
        policy_name=item["name"],
        explanation=item["description"],
        source="preset",
    )


def _deterministic_interpretation(request: str) -> PolicyInterpretation:
    """Useful offline fallback; explicit percentages override a matching metric."""
    text = (request or "").strip().lower()
    if any(word in text for word in ("privacy", "strict", "sensitive", "health")):
        result = preset("privacy_first")
    elif any(word in text for word in ("analytics", "utility", "fidelity", "model")):
        result = preset("analytics_first")
    else:
        result = preset("balanced")

    values = result.thresholds.model_dump()
    aliases = {
        "fidelity": "fidelity_min",
        "tstr": "tstr_min",
        "query": "query_equivalence_min",
        "re.?identification|reid": "reid_max",
        "membership": "membership_max",
        "attribute": "attribute_inference_max",
        "linkage": "linkage_max",
        "near.?duplicate": "near_duplicate_max",
    }
    for alias, field in aliases.items():
        match = re.search(rf"(?:{alias})[^\d]{{0,24}}(\d+(?:\.\d+)?)\s*%", text)
        if match:
            values[field] = min(1.0, max(0.0, float(match.group(1)) / 100.0))

    return PolicyInterpretation(
        thresholds=Thresholds(**values),
        policy_name=f"{result.policy_name} custom policy",
        explanation=(
            "Interpreted locally from the request. Review the numeric thresholds before running; "
            "this policy is technical configuration and does not certify legal compliance."
        ),
        source="deterministic",
    )


def interpret(request: str) -> PolicyInterpretation:
    fallback = _deterministic_interpretation(request)
    if not llm.is_enabled():
        return fallback

    defaults = Thresholds().model_dump()
    prompt = (
        "Translate this policy request into release-gate thresholds. No dataset rows or records "
        "are included. Higher *_min values are stricter utility requirements. Lower *_max values "
        "are stricter privacy requirements. Keep max_rounds from 1 to 5. Never claim HIPAA, GDPR, "
        "or other legal compliance. Return a short name and explain the trade-off.\n\n"
        f"Default thresholds: {defaults}\n"
        f"User policy request: {request[:2000]}"
    )
    settings = get_settings()
    parsed = llm.generate_structured(
        prompt,
        _LLMPolicy,
        model=settings.orchestrator_model,
        system_instruction=(
            "You configure technical privacy and utility thresholds. You never approve a release; "
            "deterministic code applies the resulting thresholds."
        ),
        temperature=0.1,
    )
    if parsed is None:
        return fallback
    values = parsed.model_dump()
    name = values.pop("policy_name")
    explanation = values.pop("explanation")
    return PolicyInterpretation(
        thresholds=Thresholds(**values),
        policy_name=name[:80],
        explanation=explanation[:500],
        source="gemini",
    )
