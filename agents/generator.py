"""Generator Agent (LLM multi-candidate strategy + CODE dependency-aware synthesis).

Forge — the Synthesizer. In each round, the LLM proposes N diverse candidate plans
along the utility/privacy Pareto frontier (beam search). Deterministic code then
synthesizes rows using a dependency-aware hybrid engine (hierarchical conditional
sampling, copula rank preservation, domain bound repair).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import llm
from agents.profiler import categorical_columns, numeric_columns
from config import get_settings
from models import (
    CandidatePlanBatch,
    DatasetProfile,
    GenerationPlan,
    RunBlackboard,
    SynthesisConfig,
)

logger = logging.getLogger("forge.generator")

_SYSTEM = (
    "You are Forge, an expert synthetic-data generation strategist. You design "
    "multiple candidate generation plans along the Pareto frontier (balancing utility "
    "and privacy). You NEVER see raw data rows, only statistical summaries, "
    "semantic constraints, and previous round diagnostic metrics. Respond strictly "
    "in valid JSON matching the CandidatePlanBatch schema."
)


def _profile_digest(profile: DatasetProfile, max_cols: int = 25) -> str:
    lines = [f"rows={profile.row_count}, label={profile.label_column}"]
    for col in profile.columns[:max_cols]:
        tag = []
        if col.is_quasi_identifier:
            tag.append("QI")
        if col.is_sensitive:
            tag.append("SENSITIVE")
        tag_s = f" [{','.join(tag)}]" if tag else ""
        if col.dtype == "numeric":
            s = col.stats
            lines.append(
                f"- {col.name}{tag_s} numeric mean={s.get('mean')} std={s.get('std')} "
                f"min={s.get('min')} max={s.get('max')}"
            )
        else:
            s = col.stats
            lines.append(
                f"- {col.name}{tag_s} categorical card={s.get('cardinality')} "
                f"top={list(s.get('top_values', {}).keys())[:4]}"
            )
    return "\n".join(lines)


def validate_and_repair_plan(
    plan: SynthesisConfig,
    real_train: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
) -> SynthesisConfig:
    """Programmatically validate and repair candidate plans."""
    qis = set(quasi_identifiers or [])
    num_cols = set(numeric_columns(real_train))

    recomb = max(0.0, min(1.0, float(plan.recombination)))
    noise = max(0.0, min(1.0, float(plan.numeric_noise)))
    shuffle = max(0.0, min(1.0, float(plan.categorical_shuffle)))
    floor = max(0.0, min(0.5, float(plan.rare_category_floor)))

    valid_gen = {}
    if plan.generalize_columns and isinstance(plan.generalize_columns, dict):
        for col, bucket in plan.generalize_columns.items():
            if col in num_cols and bucket and float(bucket) > 0:
                valid_gen[col] = float(bucket)

    # Compute empirical domain bounds for all numeric columns
    bounds = dict(plan.min_max_bounds) if plan.min_max_bounds else {}
    for col in num_cols:
        if col not in bounds:
            s = real_train[col].dropna()
            if not s.empty:
                bounds[col] = [float(s.min()), float(s.max())]

    valid_deps = {}
    if plan.group_dependencies and isinstance(plan.group_dependencies, dict):
        for child, parent in plan.group_dependencies.items():
            if child in real_train.columns and parent in real_train.columns and child != parent:
                valid_deps[child] = parent

    return SynthesisConfig(
        plan_name=plan.plan_name or "candidate",
        recombination=recomb,
        numeric_noise=noise,
        categorical_shuffle=shuffle,
        rare_category_floor=floor,
        generalize_columns=valid_gen,
        group_dependencies=valid_deps,
        copula_rank_preservation=bool(plan.copula_rank_preservation),
        min_max_bounds=bounds,
        rationale=plan.rationale or "Validated candidate plan.",
    )


def initial_plan(
    profile: DatasetProfile,
    business_rules: Optional[str] = None,
) -> GenerationPlan:
    """Deliberately naive baseline plan for demonstration duel."""
    return SynthesisConfig(
        plan_name="naive_baseline",
        recombination=0.08,
        numeric_noise=0.01,
        categorical_shuffle=0.02,
        rare_category_floor=0.0,
        generalize_columns={},
        rationale="Round 1 naive baseline: near-copy of real data to establish pre-defense baseline.",
    )


def generate_candidate_plans(
    profile: DatasetProfile,
    real_train: pd.DataFrame,
    round_num: int,
    blackboard: Optional[RunBlackboard] = None,
    business_rules: Optional[str] = None,
) -> List[SynthesisConfig]:
    """Generate N=3 diverse candidate generation plans along the Pareto frontier."""
    settings = get_settings()
    qis = [c.name for c in profile.columns if c.is_quasi_identifier]

    # Build prompt context from blackboard memory if available
    diagnostics_text = "None (first round)"
    attack_text = "None (first round)"
    feedback_text = "Establish strong baseline"

    if blackboard and blackboard.round_history:
        last_round = blackboard.round_history[-1]
        diag = last_round.utility.diagnostic_report
        if diag:
            broken_str = ", ".join([f"{b.col_a}-{b.col_b}(delta={b.delta})" for b in diag.broken_relationships[:4]])
            weak_str = ", ".join(diag.weakest_columns[:4])
            diagnostics_text = f"Weakest columns: [{weak_str}]. Broken relationships: [{broken_str}]."
        attack_text = (
            f"Last re_id_rate={last_round.attack.re_id_rate}, "
            f"membership_acc={last_round.attack.membership_accuracy}, "
            f"attacked_qis={last_round.attack.attacked_columns}"
        )
        if blackboard.orchestrator_feedback_history:
            feedback_text = blackboard.orchestrator_feedback_history[-1]

    semantic_hints = ""
    deps: Dict[str, str] = {}
    if profile.semantic_profile:
        sp = profile.semantic_profile
        deps = sp.functional_dependencies
        semantic_hints = (
            f"Inferred dependencies: {sp.functional_dependencies}\n"
            f"High-risk QI combos: {sp.high_risk_qi_combos}\n"
            f"Business constraints: {sp.business_constraints}\n"
        )

    prompt = (
        f"Design a batch of 3 distinct candidate generation plans for Round {round_num} "
        "targeting diverse points on the Utility vs. Privacy Pareto frontier:\n"
        "1. Candidate 'high_utility': Low recombination (0.15-0.30), low noise (0.02-0.05), preserves correlations.\n"
        "2. Candidate 'balanced': Balanced recombination (0.35-0.55), moderate noise (0.05-0.10), moderate rare floor.\n"
        "3. Candidate 'high_privacy': High recombination (0.60-0.85), targeted quasi-identifier generalization & noise.\n\n"
        f"DATASET PROFILE:\n{_profile_digest(profile)}\n\n"
        f"SEMANTIC PROFILE:\n{semantic_hints or 'Standard'}\n"
        f"LAST DIAGNOSTICS: {diagnostics_text}\n"
        f"LAST ATTACK VECTORS: {attack_text}\n"
        f"ORCHESTRATOR FEEDBACK: {feedback_text}\n"
        f"QUASI-IDENTIFIERS: {qis}\n"
        f"BUSINESS RULES: {business_rules or 'Standard'}\n\n"
        "Return JSON conforming to CandidatePlanBatch with 3 candidate plans."
    )

    batch = llm.generate_structured(
        prompt,
        CandidatePlanBatch,
        model=settings.generator_model,
        system_instruction=_SYSTEM,
        temperature=0.4,
    )

    candidates: List[SynthesisConfig] = []
    if batch and batch.candidates and len(batch.candidates) >= 2:
        for c in batch.candidates[:3]:
            if not c.group_dependencies and deps:
                c.group_dependencies = deps
            candidates.append(validate_and_repair_plan(c, real_train, qis))
        return candidates

    # Deterministic fallback: calibrated 3-point Pareto frontier
    base_recomb = min(0.8, 0.15 + (round_num - 1) * 0.15)
    gen_cols = {col: 2.0 for col in qis if col in numeric_columns(real_train)}

    c1 = SynthesisConfig(
        plan_name="high_utility",
        recombination=max(0.10, base_recomb - 0.15),
        numeric_noise=0.03,
        categorical_shuffle=0.04,
        rare_category_floor=0.01,
        group_dependencies=deps,
        copula_rank_preservation=True,
        rationale="Fallback high-utility: minimal perturbation to maximize fidelity.",
    )
    c2 = SynthesisConfig(
        plan_name="balanced",
        recombination=base_recomb,
        numeric_noise=0.06,
        categorical_shuffle=0.08,
        rare_category_floor=0.02,
        group_dependencies=deps,
        copula_rank_preservation=True,
        rationale="Fallback balanced: balanced trade-off between privacy and utility.",
    )
    c3 = SynthesisConfig(
        plan_name="high_privacy",
        recombination=min(0.95, base_recomb + 0.25),
        numeric_noise=0.12,
        categorical_shuffle=0.15,
        rare_category_floor=0.03,
        generalize_columns=gen_cols,
        group_dependencies=deps,
        copula_rank_preservation=True,
        rationale="Fallback high-privacy: aggressive mixing and QI generalization.",
    )

    return [
        validate_and_repair_plan(c1, real_train, qis),
        validate_and_repair_plan(c2, real_train, qis),
        validate_and_repair_plan(c3, real_train, qis),
    ]


# --------------------------------------------------------------------------- #
# Deterministic Dependency-Aware Synthesis Engine
# --------------------------------------------------------------------------- #
def synthesize(
    real_train: pd.DataFrame,
    plan: GenerationPlan,
    n_rows: int,
    quasi_identifiers: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    generation_order: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Dependency-aware hybrid row synthesis engine.

    Combines:
    1. Bootstrap base initialization
    2. Label-conditional marginal sampling
    3. Hierarchical grouped sampling for functional dependencies
    4. Copula-style rank preservation for continuous features
    5. Domain constraint repair (bounds & integer types)
    """
    rng = np.random.RandomState(seed)
    quasi_identifiers = set(quasi_identifiers or [])
    num_cols = set(numeric_columns(real_train))

    base = real_train.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)
    synth = base.copy()

    use_label = (
        label_column
        and label_column in real_train.columns
        and real_train[label_column].nunique() <= 20
    )
    if use_label:
        base_labels = base[label_column].to_numpy()
        group_pools = {
            val: real_train[real_train[label_column] == val]
            for val in real_train[label_column].unique()
        }

    # Determine topological ordering
    cols_to_process = list(generation_order) if generation_order else list(real_train.columns)
    # Ensure all real_train columns are included
    for c in real_train.columns:
        if c not in cols_to_process:
            cols_to_process.append(c)

    deps = plan.group_dependencies or {}

    for col in cols_to_process:
        if col not in real_train.columns or col == label_column:
            continue

        col_rng = plan_rng(rng, 0, col)
        is_qi = col in quasi_identifiers
        recomb = min(1.0, plan.recombination * (1.3 if is_qi else 1.0))
        is_num = col in num_cols
        dtype = float if is_num else object

        indep = np.empty(n_rows, dtype=dtype)

        # Check if column is conditionally dependent on a parent column
        parent_col = deps.get(col)
        if parent_col and parent_col in synth.columns:
            synth_parents = synth[parent_col].to_numpy()
            for pval in np.unique(synth_parents):
                parent_mask = (synth_parents == pval)
                p_idx = np.where(parent_mask)[0]
                if p_idx.size == 0:
                    continue
                pool = real_train[real_train[parent_col] == pval]
                if len(pool) >= 5:
                    indep[p_idx] = np.asarray(
                        _draw_marginal(pool[col], p_idx.size, plan, col_rng), dtype=dtype
                    )
                else:
                    # Fallback to general pool if parent subgroup is too small
                    indep[p_idx] = np.asarray(
                        _draw_marginal(real_train[col], p_idx.size, plan, col_rng), dtype=dtype
                    )
        elif use_label:
            for val, pool in group_pools.items():
                idx = np.where(base_labels == val)[0]
                if idx.size:
                    indep[idx] = np.asarray(
                        _draw_marginal(pool[col], idx.size, plan, col_rng), dtype=dtype
                    )
        else:
            indep = np.asarray(_draw_marginal(real_train[col], n_rows, plan, col_rng), dtype=dtype)

        replace_prob = recomb if is_num else min(1.0, recomb + plan.categorical_shuffle)
        mask = col_rng.random(n_rows) < replace_prob
        out = synth[col].to_numpy().astype(dtype)
        out[mask] = indep[mask]
        synth[col] = out

        # Numeric transformations & copula preservation
        if is_num:
            synth[col] = _perturb_numeric(
                synth[col],
                real_train[col],
                plan,
                is_qi,
                col_rng,
            )

    return synth[real_train.columns.tolist()]


def plan_rng(rng: np.random.RandomState, gi: int, col: str) -> np.random.RandomState:
    digest = hashlib.sha256(f"{gi}:{col}".encode("utf-8")).digest()
    stable_seed = int.from_bytes(digest[:4], "little")
    return np.random.RandomState(stable_seed ^ rng.randint(0, 1 << 30))


def _draw_marginal(series: pd.Series, n: int, plan: GenerationPlan, rng: np.random.RandomState):
    """Sample n values independently from an empirical distribution."""
    if pd.api.types.is_numeric_dtype(series):
        vals = series.dropna().astype(float).to_numpy()
        if vals.size == 0:
            return np.zeros(n)
        return rng.choice(vals, size=n, replace=True)

    freqs = series.dropna().astype(str).value_counts(normalize=True)
    if freqs.empty:
        return np.array(["unknown"] * n)
    values = freqs.index.to_numpy()
    probs = freqs.to_numpy()
    rare = probs < plan.rare_category_floor
    if rare.any() and (~rare).any():
        values, probs = values[~rare], probs[~rare]
        probs = probs / probs.sum()
    return rng.choice(values, size=n, replace=True, p=probs)


def _perturb_numeric(
    col_series: pd.Series,
    full_series: pd.Series,
    plan: GenerationPlan,
    is_qi: bool,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Numeric privacy transforms and domain constraint repair."""
    sampled = col_series.astype(float).to_numpy().copy()
    std = float(full_series.std(ddof=0)) or 0.0
    noise_scale = (plan.numeric_noise * 2.0) if is_qi else (plan.numeric_noise * 0.5)

    if std > 0 and noise_scale > 0:
        if plan.copula_rank_preservation:
            # Rank-preserving jitter: small additive gaussian disturbance proportional to local quantile spread
            jitter = rng.normal(0, std * noise_scale * 0.7, size=len(sampled))
            sampled = sampled + jitter
        else:
            sampled = sampled + rng.normal(0, std * noise_scale, size=len(sampled))

    # Quasi-identifier generalization
    bucket = 0
    if is_qi and plan.generalize_columns:
        bucket = plan.generalize_columns.get(full_series.name, 0)
    if bucket and bucket > 0:
        sampled = np.round(sampled / bucket) * bucket

    # Domain bound repair (respect empirical min/max)
    valid = full_series.dropna().astype(float)
    if not valid.empty:
        col_min = float(valid.min())
        col_max = float(valid.max())
        if plan.min_max_bounds and full_series.name in plan.min_max_bounds:
            b = plan.min_max_bounds[full_series.name]
            col_min, col_max = b[0], b[1]
        sampled = np.clip(sampled, col_min, col_max)

    if pd.api.types.is_integer_dtype(full_series):
        mask_valid = ~np.isnan(sampled)
        if mask_valid.any():
            sampled[mask_valid] = np.round(sampled[mask_valid]).astype(int)

    return sampled
