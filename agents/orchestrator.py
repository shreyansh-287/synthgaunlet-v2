"""Orchestrator (CODE decision gate + Pareto Selector + LLM strategy).

The Arbiter. A deterministic release gate decides pass/fail from metrics.
Multi-candidate Pareto selection picks the optimal candidate from the frontier.
When rounds fail, the Arbiter LLM reviews the RunBlackboard (diagnostics + attacks)
to direct the Generator's next exploration.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import llm
from config import get_settings
from models import (
    AttackResult,
    CandidateEvaluation,
    GenerationPlan,
    RunBlackboard,
    Thresholds,
    UtilityResult,
)

logger = logging.getLogger("forge.orchestrator")

_SYSTEM = (
    "You are the Arbiter orchestrating an adversarial synthetic-data loop. Given "
    "utility and privacy metrics, broken relationships, and attack vectors, you diagnose "
    "the failure root-cause and instruct Forge on how to adjust parameters. "
    "Respond with a single JSON object."
)


def decide(utility: UtilityResult, attack: AttackResult, t: Thresholds) -> bool:
    """100% deterministic release gate. Pure code, no LLM."""
    return (
        utility.fidelity_score >= t.fidelity_min
        and utility.utility_ratio >= t.tstr_min
        and utility.query_equivalence >= t.query_equivalence_min
        and attack.re_id_rate <= t.reid_max
        and attack.membership_accuracy <= t.membership_max
    )


def select_pareto_candidate(
    evaluations: List[CandidateEvaluation],
    t: Thresholds,
) -> int:
    """Deterministic Pareto selection: evaluates all generated candidates per round.

    1. Prioritize candidates that satisfy hard privacy constraints.
    2. Among privacy-compliant candidates, pick highest composite utility.
    3. If none pass privacy, pick candidate closest to passing with acceptable fidelity.
    """
    if not evaluations:
        return 0

    # Calculate composite utility and compliance
    for i, e in enumerate(evaluations):
        comp_util = (
            e.utility.fidelity_score * 0.40
            + e.utility.utility_ratio * 0.40
            + e.utility.query_equivalence * 0.20
        )
        priv_passed = (
            e.attack.re_id_rate <= t.reid_max
            and e.attack.membership_accuracy <= t.membership_max
        )
        util_passed = (
            e.utility.fidelity_score >= t.fidelity_min
            and e.utility.utility_ratio >= t.tstr_min
            and e.utility.query_equivalence >= t.query_equivalence_min
        )

        if priv_passed and util_passed:
            score = 100.0 + comp_util
        elif priv_passed:
            score = 50.0 + comp_util
        else:
            # Penalize privacy leakage heavily
            reid_excess = max(0.0, e.attack.re_id_rate - t.reid_max)
            mem_excess = max(0.0, e.attack.membership_accuracy - t.membership_max)
            penalty = (reid_excess * 100.0) + (mem_excess * 20.0)
            score = comp_util - penalty

        evaluations[i].composite_score = round(float(score), 4)

    # Pick candidate with highest composite score
    best_idx = 0
    best_score = evaluations[0].composite_score
    for i, e in enumerate(evaluations):
        if e.composite_score > best_score:
            best_score = e.composite_score
            best_idx = i

    return best_idx


def diagnose_and_advise(
    blackboard: RunBlackboard,
    t: Thresholds,
) -> str:
    """Arbiter LLM diagnosis using the full RunBlackboard memory."""
    settings = get_settings()
    if not blackboard.round_history:
        return "Initial round: calibrate baseline across Pareto candidates."

    last_round = blackboard.round_history[-1]
    u = last_round.utility
    a = last_round.attack
    diag = u.diagnostic_report

    weakest_str = ", ".join(diag.weakest_columns[:5]) if diag and diag.weakest_columns else "None"
    broken_str = (
        ", ".join([f"{b.col_a}-{b.col_b} (delta={b.delta})" for b in diag.broken_relationships[:4]])
        if diag and diag.broken_relationships
        else "None"
    )

    prompt = (
        "The winning candidate FAILED the release gate. Diagnose the root cause and advise Forge.\n\n"
        f"Round {last_round.round_index} metrics:\n"
        f"- Fidelity: {u.fidelity_score} (target >={t.fidelity_min})\n"
        f"- TSTR Utility: {u.utility_ratio} (target >={t.tstr_min})\n"
        f"- Query Equivalence: {u.query_equivalence} (target >={t.query_equivalence_min})\n"
        f"- Re-id Rate: {a.re_id_rate} (target <={t.reid_max})\n"
        f"- Membership Accuracy: {a.membership_accuracy} (target <={t.membership_max})\n"
        f"- Outlier Linkage: {a.outlier_linkage_rate}\n"
        f"- Weakest Columns: [{weakest_str}]\n"
        f"- Broken Relationships: [{broken_str}]\n"
        f"- Attacker Strategy: {a.attacker_strategy}\n\n"
        "Provide a concise, 1-2 sentence actionable instruction for Forge's next candidate batch:\n"
        '{"feedback": "<instruction to Forge>", "priority": "privacy|utility|balanced"}'
    )

    data = llm.generate_json(prompt, model=settings.orchestrator_model, system_instruction=_SYSTEM)
    if data and data.get("feedback"):
        return str(data.get("feedback"))

    # Rule-based fallback
    if a.re_id_rate > t.reid_max or a.membership_accuracy > t.membership_max:
        return (
            f"Privacy leaking on {a.attacked_columns}: increase recombination and apply "
            "generalization to quasi-identifiers."
        )
    if u.utility_ratio < t.tstr_min or u.fidelity_score < t.fidelity_min:
        return (
            f"Utility dropped on [{weakest_str}]: preserve copula rank ordering and reduce noise on predictive features."
        )
    return "Balanced tuning required: nudge recombination while protecting correlation structures."


def next_plan(
    plan: GenerationPlan,
    utility: UtilityResult,
    attack: AttackResult,
    t: Thresholds,
) -> Tuple[GenerationPlan, str]:
    """Single-candidate adjustment fallback for legacy test compat."""
    settings = get_settings()
    prompt = (
        "The synthetic data FAILED the gate. Adjust the generator knobs.\n\n"
        f"Thresholds: fidelity>={t.fidelity_min}, TSTR>={t.tstr_min}, "
        f"query_equivalence>={t.query_equivalence_min}, re_id<={t.reid_max}, "
        f"membership<={t.membership_max}\n"
        f"Current metrics: fidelity={utility.fidelity_score}, "
        f"TSTR={utility.utility_ratio}, query_equivalence={utility.query_equivalence}, "
        f"re_id={attack.re_id_rate}, "
        f"membership={attack.membership_accuracy}, "
        f"outlier_linkage={attack.outlier_linkage_rate}\n"
        f"Current knobs: recombination={plan.recombination}, "
        f"numeric_noise={plan.numeric_noise}, "
        f"categorical_shuffle={plan.categorical_shuffle}, "
        f"rare_category_floor={plan.rare_category_floor}\n\n"
        "'recombination' is the PRIMARY dial: higher = each row mixes more people "
        "= less leakage. If privacy (re_id/membership) is failing, INCREASE "
        "recombination first (by ~0.2-0.3), then noise/rare_floor. If fidelity is "
        "failing, DECREASE recombination/noise. Keep values in (0..1).\n"
        '{"recombination": <0..1>, "numeric_noise": <0..1>, '
        '"categorical_shuffle": <0..1>, "rare_category_floor": <0..1>, '
        '"feedback": "<one sentence to Forge>"}'
    )
    data = llm.generate_json(prompt, model=settings.orchestrator_model, system_instruction=_SYSTEM)
    if data:
        try:
            new_plan = GenerationPlan(
                recombination=_clip(data.get("recombination", plan.recombination)),
                numeric_noise=_clip(data.get("numeric_noise", plan.numeric_noise)),
                categorical_shuffle=_clip(data.get("categorical_shuffle", plan.categorical_shuffle)),
                rare_category_floor=_clip(data.get("rare_category_floor", plan.rare_category_floor)),
                generalize_columns=plan.generalize_columns,
                rationale=data.get("feedback", ""),
            )
            if _privacy_failing(attack, t):
                new_plan.generalize_columns.update({column: 2.0 for column in attack.attacked_columns})
                if new_plan.recombination <= plan.recombination:
                    new_plan.recombination = _clip(plan.recombination + 0.25)
            return new_plan, data.get("feedback", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Orchestrator LLM plan invalid, using rule-based: %s", exc)

    return _rule_based(plan, utility, attack, t)


def _privacy_failing(attack: AttackResult, t: Thresholds) -> bool:
    return attack.re_id_rate > t.reid_max or attack.membership_accuracy > t.membership_max


def _utility_failing(utility: UtilityResult, t: Thresholds) -> bool:
    return (
        utility.fidelity_score < t.fidelity_min
        or utility.utility_ratio < t.tstr_min
        or utility.query_equivalence < t.query_equivalence_min
    )


def _rule_based(plan, utility, attack, t) -> Tuple[GenerationPlan, str]:
    privacy_failing = attack.re_id_rate > t.reid_max or attack.membership_accuracy > t.membership_max
    utility_failing = _utility_failing(utility, t)

    recombination = plan.recombination
    numeric_noise = plan.numeric_noise
    categorical_shuffle = plan.categorical_shuffle
    rare_floor = plan.rare_category_floor
    generalize_columns = dict(plan.generalize_columns)
    if privacy_failing:
        generalize_columns.update({column: 2.0 for column in attack.attacked_columns})

    if privacy_failing and not utility_failing:
        recombination = _clip(plan.recombination + 0.3)
        numeric_noise = _clip(plan.numeric_noise + 0.04)
        rare_floor = _clip(plan.rare_category_floor + 0.01)
        feedback = ("Privacy still leaking: mixing each row across more people "
                    "(recombination up) plus extra quasi-identifier jitter.")
    elif utility_failing and not privacy_failing:
        recombination = _clip(plan.recombination - 0.15)
        numeric_noise = _clip(plan.numeric_noise - 0.02)
        feedback = "Utility dipped: keeping rows closer to real to sharpen the data."
    else:
        recombination = _clip(plan.recombination + 0.15)
        numeric_noise = _clip(plan.numeric_noise + 0.02)
        feedback = "Both under pressure: nudging privacy up while watching fidelity."

    new_plan = GenerationPlan(
        recombination=recombination,
        numeric_noise=numeric_noise,
        categorical_shuffle=categorical_shuffle,
        rare_category_floor=rare_floor,
        generalize_columns=generalize_columns,
        rationale=feedback,
    )
    return new_plan, feedback


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return float(max(lo, min(hi, float(v))))
    except (TypeError, ValueError):
        return lo
