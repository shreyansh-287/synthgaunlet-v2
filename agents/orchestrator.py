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


def _privacy_failing(attack: AttackResult, t: Thresholds) -> bool:
    return (
        attack.re_id_rate > t.reid_max
        or attack.membership_accuracy > t.membership_max
        or attack.attribute_inference_advantage > t.attribute_inference_max
        or attack.linkage_rate > t.linkage_max
        or attack.near_duplicate_rate > t.near_duplicate_max
    )


def decide(utility: UtilityResult, attack: AttackResult, t: Thresholds) -> bool:
    """100% deterministic release gate. Pure code, no LLM."""
    return (
        utility.fidelity_score >= t.fidelity_min
        and utility.utility_ratio >= t.tstr_min
        and utility.query_equivalence >= t.query_equivalence_min
        and not _privacy_failing(attack, t)
    )


def outcome_score(utility: UtilityResult, attack: AttackResult, t: Thresholds) -> float:
    """Deterministic rank used to pick the best failed candidate for rescue."""
    comp_util = (
        utility.fidelity_score * 0.40
        + utility.utility_ratio * 0.40
        + utility.query_equivalence * 0.20
    )
    priv_passed = not _privacy_failing(attack, t)
    util_passed = (
        utility.fidelity_score >= t.fidelity_min
        and utility.utility_ratio >= t.tstr_min
        and utility.query_equivalence >= t.query_equivalence_min
    )
    if priv_passed and util_passed:
        score = 100.0 + comp_util
    elif priv_passed:
        score = 50.0 + comp_util
    else:
        reid_excess = max(0.0, attack.re_id_rate - t.reid_max)
        mem_excess = max(0.0, attack.membership_accuracy - t.membership_max)
        attribute_excess = max(0.0, attack.attribute_inference_advantage - t.attribute_inference_max)
        linkage_excess = max(0.0, attack.linkage_rate - t.linkage_max)
        duplicate_excess = max(0.0, attack.near_duplicate_rate - t.near_duplicate_max)
        penalty = (
            (reid_excess * 100.0)
            + (mem_excess * 20.0)
            + (attribute_excess * 40.0)
            + (linkage_excess * 40.0)
            + (duplicate_excess * 40.0)
        )
        score = comp_util - penalty
    return round(float(score), 4)


def build_rescue_plan(
    attempt: int,
    plan: GenerationPlan,
    numeric_quasi_identifiers: Optional[List[str]] = None,
) -> GenerationPlan:
    """Knob settings for one Auto-Heal attempt. Row repair stays in the generator."""
    numeric_qis = list(numeric_quasi_identifiers or [])
    if attempt == 1:
        return plan.model_copy(update={
            "plan_name": "rescue_tighten",
            "recombination": max(0.65, float(plan.recombination)),
            "numeric_noise": min(1.0, float(plan.numeric_noise) * 1.5),
            "rare_category_floor": max(0.05, float(plan.rare_category_floor)),
            "rationale": (
                "Rescue attempt 1: recombination at least 0.65, jitter up 50%, "
                "rare-category floor 0.05, then drop the closest DCR rows."
            ),
        })
    if attempt == 2:
        bands = dict(plan.generalize_columns)
        for col in numeric_qis:
            bands[col] = 5.0 if "age" in col.lower() else max(float(bands.get(col, 0.0)), 1.0)
        return plan.model_copy(update={
            "plan_name": "rescue_generalize",
            "generalize_columns": bands,
            "rare_category_floor": max(0.02, float(plan.rare_category_floor)),
            "rationale": (
                "Rescue attempt 2: bucket numeric quasi-identifiers, collapse "
                "categories under 2% to Other, and winsorize the 1st/99th percentiles."
            ),
        })
    return plan.model_copy(update={
        "plan_name": "rescue_second_generation",
        "recombination": min(1.0, max(0.85, float(plan.recombination) + 0.2)),
        "numeric_noise": min(1.0, float(plan.numeric_noise) * 1.5 + 0.05),
        "categorical_shuffle": min(1.0, float(plan.categorical_shuffle) + 0.15),
        "rare_category_floor": max(0.05, float(plan.rare_category_floor)),
        "copula_rank_preservation": False,
        "rationale": (
            "Rescue attempt 3: sample a new table from the best failed candidate "
            "with stronger perturbation."
        ),
    })


def utility_score(utility: UtilityResult) -> float:
    """Mean of fidelity, query equivalence, and train-on-synthetic utility."""
    return round(
        (
            float(utility.fidelity_score)
            + float(utility.query_equivalence)
            + float(utility.utility_ratio)
        ) / 3.0,
        6,
    )


def _risk_ratio(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0 if value <= 0 else 10.0
    return max(0.0, float(value) / float(limit))


def privacy_risk(attack: AttackResult, t: Thresholds) -> float:
    """Worst normalized privacy signal. 1.0 sits on the release ceiling.

    Re-identification, attribute inference, and linkage are value/limit.
    Membership is the excess over a 0.5 chance baseline, divided by the
    gap from 0.5 up to the membership ceiling. Near-duplicate stays on the
    release gate and is not part of this axis.
    """
    span = float(t.membership_max) - 0.5
    if span <= 0:
        membership = 0.0 if float(attack.membership_accuracy) <= float(t.membership_max) else 10.0
    else:
        membership = max(0.0, float(attack.membership_accuracy) - 0.5) / span
    return round(max(
        _risk_ratio(attack.re_id_rate, t.reid_max),
        membership,
        _risk_ratio(attack.attribute_inference_advantage, t.attribute_inference_max),
        _risk_ratio(attack.linkage_rate, t.linkage_max),
    ), 6)


def score_candidate(
    candidate_index: int,
    plan: GenerationPlan,
    utility: UtilityResult,
    attack: AttackResult,
    t: Thresholds,
    round_index: int,
) -> CandidateEvaluation:
    """Attach utility, privacy risk, and the release-gate flag. Frontier comes later."""
    passed = decide(utility, attack, t)
    return CandidateEvaluation(
        candidate_index=candidate_index,
        candidate_id=f"r{round_index}-c{candidate_index}",
        plan=plan,
        utility=utility,
        attack=attack,
        passed=passed,
        gate_passed=passed,
        utility_score=utility_score(utility),
        privacy_risk=privacy_risk(attack, t),
    )


def mark_pareto(evaluations: List[CandidateEvaluation]) -> None:
    """Flag the 2D skyline. A dominates B when utility is at least as high and
    privacy risk is at most as high, with at least one of those strict.
    """
    for candidate in evaluations:
        dominated = False
        for other in evaluations:
            if other is candidate:
                continue
            as_good = (
                other.utility_score >= candidate.utility_score - 1e-9
                and other.privacy_risk <= candidate.privacy_risk + 1e-9
            )
            strictly_better = (
                other.utility_score > candidate.utility_score + 1e-9
                or other.privacy_risk < candidate.privacy_risk - 1e-9
            )
            if as_good and strictly_better:
                dominated = True
                break
        candidate.is_pareto_optimal = not dominated


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

    for i, e in enumerate(evaluations):
        evaluations[i].composite_score = outcome_score(e.utility, e.attack, t)

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
    if _privacy_failing(a, t):
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


def _utility_failing(utility: UtilityResult, t: Thresholds) -> bool:
    return (
        utility.fidelity_score < t.fidelity_min
        or utility.utility_ratio < t.tstr_min
        or utility.query_equivalence < t.query_equivalence_min
    )


def _rule_based(plan, utility, attack, t) -> Tuple[GenerationPlan, str]:
    privacy_failing = _privacy_failing(attack, t)
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
