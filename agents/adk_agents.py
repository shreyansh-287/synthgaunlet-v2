"""Google ADK agent definitions for the SynthGauntlet roles."""

from __future__ import annotations

from typing import Any, Dict

from config import get_settings
from models import AuditMetrics, DatasetProfile, PrivacyAttackResults, SynthesisConfig

try:
    from google.adk.agents.llm_agent import Agent
except ImportError:
    Agent = None


def _profile_for_agent(profile: dict) -> dict:
    """Validate and return aggregate-only profile data for an ADK agent."""
    value = DatasetProfile.model_validate(profile)
    return value.model_dump(exclude_none=True)


def forge_profile_tool(profile: dict, feedback: str = "") -> dict:
    """Return schema statistics for Forge without exposing source rows."""
    return {"profile": _profile_for_agent(profile), "feedback": feedback[:2000]}


def challenger_metrics_tool(metrics: dict) -> dict:
    """Validate sanitized deterministic attack output for Challenger."""
    return PrivacyAttackResults.model_validate(metrics).model_dump()


def arbiter_gate_tool(utility: dict, privacy: dict, thresholds: dict) -> dict:
    """Validate metric contracts before the deterministic release gate."""
    audit = AuditMetrics.model_validate(utility)
    attack = PrivacyAttackResults.model_validate(privacy)
    return {
        "utility": audit.model_dump(),
        "privacy": attack.model_dump(),
        "thresholds": thresholds,
    }


def _agent(name: str, model: str, description: str, instruction: str, tools: list):
    if Agent is None:
        return {
            "name": name,
            "description": description,
            "instruction": instruction,
            "tools": [tool.__name__ for tool in tools],
        }
    return Agent(
        name=name,
        model=model,
        description=description,
        instruction=instruction,
        tools=tools,
    )


def adk_available() -> bool:
    return Agent is not None


def build_agents() -> Dict[str, Any]:
    """Build Forge, Challenger, and Arbiter with Gemini-safe tool boundaries."""
    return {
        "forge": _agent(
            "forge", get_settings().generator_model,
            "Chooses deterministic tabular synthesis parameters.",
            "Use only aggregate DatasetProfile data. Choose SynthesisConfig parameters; never request or output source rows.",
            [forge_profile_tool],
        ),
        "challenger": _agent(
            "challenger", get_settings().attacker_model,
            "Plans privacy attacks against a synthetic dataset.",
            "Use only DatasetProfile and sanitized deterministic tool results. Never request, inspect, or reproduce source rows.",
            [challenger_metrics_tool],
        ),
        "arbiter": _agent(
            "arbiter", get_settings().orchestrator_model,
            "Explains deterministic release decisions.",
            "Treat deterministic metrics and thresholds as authoritative. Never override the code release gate and never request source rows.",
            [arbiter_gate_tool],
        ),
    }


AGENTS = build_agents()
