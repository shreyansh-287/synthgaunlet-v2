"""Attacker Agent (Challenger - Privacy Attacker).

Challenger (Privacy Attacker): uses a bounded function-calling loop
(max 3 iterations) to probe synthetic tabular data for privacy leaks, and reports
the worst-case (maximum risk) breach metrics across all attempts to protect the release gate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

import llm
from agents import attack_tools
from agents.attacks.registry import run_attacks, extended_attacks
from config import get_settings
from models import AttackResult, DatasetProfile

logger = logging.getLogger("forge.attacker")

_SYSTEM = (
    "You are Challenger, a relentless ethical privacy attacker. Your goal is to break "
    "the privacy of synthetic tabular data. You reason about where it is weakest "
    "and choose attack parameters. Respond with a single JSON object only."
)

_NARRATOR = (
    "You are Challenger, an ethical privacy attacker writing a short, candid field "
    "report in plain prose. Do NOT use JSON or markdown. 2-3 sentences, first person."
)


def _plan_initial_attack(
    profile: DatasetProfile,
    quasi_identifiers: List[str],
    prior: Optional[Dict],
    default_threshold: float,
) -> Dict[str, Any]:
    settings = get_settings()
    prior_txt = "none (first round)"
    if prior:
        prior_txt = (
            f"last re_id_rate={prior.get('re_id_rate')}, "
            f"membership_accuracy={prior.get('membership_accuracy')}, "
            f"generator_feedback={prior.get('generator_feedback')}"
        )

    # Inferred high risk combos from semantic profile if available
    semantic_hints = ""
    if profile.semantic_profile and profile.semantic_profile.high_risk_qi_combos:
        semantic_hints = f"High-risk QI combos: {profile.semantic_profile.high_risk_qi_combos}\n"

    prompt = (
        "Plan this round's attack on the synthetic data.\n"
        f"Quasi-identifiers available: {quasi_identifiers}\n"
        f"Sensitive columns: {[c.name for c in profile.columns if c.is_sensitive]}\n"
        f"{semantic_hints}"
        f"Prior round: {prior_txt}\n\n"
        "Decide:\n"
        "- which quasi-identifiers to focus the re-identification attack on\n"
        "- a distance threshold in (0.02..0.3); LOWER = stricter near-copy test\n"
        "- whether to run outlier linkage on rare combinations\n"
        "- a one-sentence hypothesis about the weakest point\n\n"
        '{"focus_columns": ["..."], "threshold": <float>, '
        '"run_outlier_linkage": <bool>, "hypothesis": "<text>"}'
    )
    data = llm.generate_json(
        prompt, model=settings.attacker_model, system_instruction=_SYSTEM,
        role="attacker", purpose="Choose which quasi-identifiers to probe and how strict the distance cutoff should be.",
    )
    if not data:
        return {
            "focus_columns": quasi_identifiers,
            "threshold": default_threshold,
            "run_outlier_linkage": True,
            "hypothesis": "Fallback: probe all quasi-identifiers for near-copies and rare combos.",
        }
    focus = [c for c in data.get("focus_columns", []) if c in quasi_identifiers]
    return {
        "focus_columns": focus or quasi_identifiers,
        "threshold": float(data.get("threshold", default_threshold) or default_threshold),
        "run_outlier_linkage": bool(data.get("run_outlier_linkage", True)),
        "hypothesis": str(data.get("hypothesis", "")),
    }


def _plan_followup_attack(
    intermediate_results: Dict[str, Any],
    quasi_identifiers: List[str],
) -> Optional[Dict[str, Any]]:
    """Determine whether to run an adaptive secondary attack pass."""
    settings = get_settings()
    prompt = (
        "Pass 1 attack results:\n"
        f"re_id_rate: {intermediate_results.get('re_id_rate')}\n"
        f"membership_accuracy: {intermediate_results.get('membership_accuracy')}\n"
        f"outlier_linkage: {intermediate_results.get('outlier_linkage_rate')}\n"
        f"Tested QIs: {intermediate_results.get('focus_columns')}\n"
        f"All QIs available: {quasi_identifiers}\n\n"
        "Decide if you want a second attack pass (e.g. probing a tighter QI subset or stricter threshold):\n"
        '{"execute_secondary": <bool>, "secondary_qis": ["..."], "stricter_threshold": <float>, "hypothesis": "<text>"}'
    )
    data = llm.generate_json(
        prompt, model=settings.attacker_model, system_instruction=_SYSTEM,
        role="attacker", purpose="Decide whether a second privacy-attack pass is useful.",
    )
    if data and data.get("execute_secondary"):
        sec_qis = [c for c in data.get("secondary_qis", []) if c in quasi_identifiers]
        return {
            "focus_columns": sec_qis or quasi_identifiers,
            "threshold": float(data.get("stricter_threshold", 0.05) or 0.05),
            "hypothesis": str(data.get("hypothesis", "Secondary focused probe.")),
        }
    return None


def _interpret(
    initial_plan: Dict[str, Any],
    worst_reid: float,
    worst_membership: float,
    worst_outlier: float,
    passes_count: int,
) -> str:
    settings = get_settings()
    facts = (
        f"hypothesis: {initial_plan.get('hypothesis')}\n"
        f"attack_passes: {passes_count}\n"
        f"worst_re_id_rate: {worst_reid}\n"
        f"worst_membership_accuracy: {worst_membership}\n"
        f"worst_outlier_linkage: {worst_outlier}"
    )
    text = llm.generate_text(
        "Given your attack results, write 2-3 sentences (first person, as Challenger) "
        "explaining what you tried across your attack passes, what leaked, and what you'll target next.\n\n"
        f"{facts}",
        model=settings.attacker_model,
        system_instruction=_NARRATOR,
        temperature=0.8,
        role="attacker",
        purpose="Write the attack narrative from the measured privacy scores.",
    )
    if text:
        return text

    verdict = "found near-copies" if worst_reid > 0.01 else "found few near-copies"
    return (
        f"I probed across {passes_count} attack passes and {verdict} "
        f"(worst re-id {worst_reid}). Peak membership signal was "
        f"{worst_membership}. I will continue targeting the rarest combinations."
    )


def attack(
    profile: DatasetProfile,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth: pd.DataFrame,
    quasi_identifiers: List[str],
    prior: Optional[Dict] = None,
) -> AttackResult:
    """Adaptive bounded attack tool loop returning worst-case privacy risk."""
    settings = get_settings()
    initial_plan = _plan_initial_attack(profile, quasi_identifiers, prior, settings.reid_threshold)

    tools_used: List[str] = ["run_reidentification", "run_membership_inference"]
    attacked_columns: List[str] = list(initial_plan["focus_columns"])

    # Pass 1: Primary attack pass
    reid_p1 = attack_tools.run_reidentification(
        real_train,
        synth,
        initial_plan["focus_columns"],
        initial_plan["threshold"],
        real_holdout=real_test,
    )

    feature_cols = [c.name for c in profile.columns if c.name in synth.columns]
    membership_p1 = attack_tools.run_membership_inference(
        real_train, real_test, synth, feature_cols
    )

    outlier_p1 = {"outlier_linkage_rate": 0.0}
    if initial_plan.get("run_outlier_linkage"):
        outlier_p1 = attack_tools.run_outlier_linkage(real_train, synth, initial_plan["focus_columns"])
        tools_used.append("run_outlier_linkage")

    all_passes = [
        {
            "re_id_rate": float(reid_p1.get("re_id_rate", 0.0)),
            "membership_accuracy": float(membership_p1.get("membership_accuracy", 0.5)),
            "outlier_linkage_rate": float(outlier_p1.get("outlier_linkage_rate", 0.0)),
            "flagged": reid_p1.get("flagged_examples", []),
        }
    ]

    # Pass 2: Adaptive follow-up probe if LLM is active
    if llm.is_enabled():
        inter_summary = {
            "re_id_rate": reid_p1.get("re_id_rate", 0.0),
            "membership_accuracy": membership_p1.get("membership_accuracy", 0.5),
            "outlier_linkage_rate": outlier_p1.get("outlier_linkage_rate", 0.0),
            "focus_columns": initial_plan["focus_columns"],
        }
        followup = _plan_followup_attack(inter_summary, quasi_identifiers)
        if followup:
            tools_used.append("run_reidentification_adaptive")
            for c in followup["focus_columns"]:
                if c not in attacked_columns:
                    attacked_columns.append(c)

            reid_p2 = attack_tools.run_reidentification(
                real_train,
                synth,
                followup["focus_columns"],
                followup["threshold"],
                real_holdout=real_test,
            )
            all_passes.append(
                {
                    "re_id_rate": float(reid_p2.get("re_id_rate", 0.0)),
                    "membership_accuracy": float(membership_p1.get("membership_accuracy", 0.5)),
                    "outlier_linkage_rate": float(outlier_p1.get("outlier_linkage_rate", 0.0)),
                    "flagged": reid_p2.get("flagged_examples", []),
                }
            )

    # Record WORST-CASE (maximum risk) observed across all attack passes
    extended = run_attacks(extended_attacks(), real_train, real_test, synth, profile)
    tools_used.extend(result.name for result in extended.values())

    worst_re_id = max(p["re_id_rate"] for p in all_passes)
    worst_membership = max(p["membership_accuracy"] for p in all_passes)
    worst_outlier = max(p["outlier_linkage_rate"] for p in all_passes)
    attribute_advantage = float(extended["attribute_inference"].score)
    linkage_rate = float(extended["linkage"].score)
    near_duplicate_rate = float(extended["near_duplicate"].score)

    flagged_examples = []
    for p in all_passes:
        flagged_examples.extend(p.get("flagged", []))

    strategy = _interpret(
        initial_plan,
        round(worst_re_id, 4),
        round(worst_membership, 4),
        round(worst_outlier, 4),
        len(all_passes),
    )

    return AttackResult(
        re_id_rate=round(worst_re_id, 4),
        membership_accuracy=round(worst_membership, 4),
        outlier_linkage_rate=round(worst_outlier, 4),
        attribute_inference_advantage=round(attribute_advantage, 4),
        linkage_rate=round(linkage_rate, 4),
        near_duplicate_rate=round(near_duplicate_rate, 4),
        flagged_examples=flagged_examples[:5],
        attacker_strategy=strategy,
        tools_used=list(dict.fromkeys(tools_used)),
        attacked_columns=attacked_columns,
    )
