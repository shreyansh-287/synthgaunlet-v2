"""Tests for the Enterprise Vertex AI & High-Convergence Multi-Candidate Architecture."""

import numpy as np
import pandas as pd
import pytest

from agents import attacker, generator, orchestrator, profiler, statistician
from models import (
    AuditMetrics,
    CandidateEvaluation,
    ColumnProfile,
    DatasetProfile,
    PrivacyAttackResults,
    RoundResult,
    RunBlackboard,
    SemanticProfile,
    SynthesisConfig,
    Thresholds,
)


def test_semantic_profile_fallback():
    """Verify that profiler provides a valid SemanticProfile even without LLM."""
    df = pd.DataFrame({
        "age": [25, 30, 45, 50, 22],
        "department": ["Sales", "HR", "Eng", "Sales", "HR"],
        "job_role": ["Rep", "Lead", "Dev", "Rep", "Lead"],
        "attrition": [0, 1, 0, 0, 1],
    })
    profile = profiler.profile_dataframe(
        df,
        quasi_identifiers=["age", "department"],
        sensitive_columns=["job_role"],
        label_column="attrition",
    )
    semantic = profiler.analyze_profile(profile)
    assert isinstance(semantic, SemanticProfile)
    assert len(semantic.conditional_generation_order) == 4
    assert semantic.conditional_generation_order[-1] == "attrition"


def test_candidate_plan_validation_and_repair():
    """Verify that candidate plans are clamped to safe parameters and valid column sets."""
    df = pd.DataFrame({"age": [20, 30, 40], "income": [2000, 3000, 4000]})
    malformed = SynthesisConfig(
        plan_name="broken_plan",
        recombination=1.5,
        numeric_noise=-0.2,
        categorical_shuffle=2.0,
        rare_category_floor=0.8,
        generalize_columns={"nonexistent_col": 5.0, "age": -3.0, "income": 500.0},
    )
    repaired = generator.validate_and_repair_plan(malformed, df, quasi_identifiers=["income"])
    assert repaired.recombination == 1.0
    assert repaired.numeric_noise == 0.0
    assert repaired.categorical_shuffle == 1.0
    assert repaired.rare_category_floor == 0.5
    assert "income" in repaired.generalize_columns
    assert "nonexistent_col" not in repaired.generalize_columns
    assert "age" not in repaired.generalize_columns


def test_dependency_aware_synthesis_bounds():
    """Verify that dependency-aware synthesis preserves empirical min/max bounds."""
    df = pd.DataFrame({
        "dept": ["IT", "IT", "HR", "HR", "Sales"],
        "role": ["Dev", "Dev", "Admin", "Admin", "Rep"],
        "salary": [5000, 6000, 4000, 4500, 5500],
        "label": [1, 0, 0, 1, 1],
    })
    plan = SynthesisConfig(
        plan_name="test_dep",
        recombination=0.5,
        group_dependencies={"role": "dept"},
        copula_rank_preservation=True,
    )
    synth = generator.synthesize(
        df, plan, n_rows=20, quasi_identifiers=["dept"], label_column="label"
    )
    assert len(synth) == 20
    assert synth["salary"].min() >= 4000
    assert synth["salary"].max() <= 6000


def test_statistician_broken_relationships_and_diagnostic():
    """Verify that deep diagnostic report identifies degraded correlations."""
    np.random.seed(42)
    x = np.linspace(10, 100, 100)
    y = x * 2.0 + np.random.normal(0, 5, 100)
    real_df = pd.DataFrame({"x": x, "y": y, "label": [0, 1] * 50})

    # Synthetic has destroyed relationship
    synth_df = pd.DataFrame({"x": x, "y": np.random.permutation(y), "label": [0, 1] * 50})

    eval_res = statistician.evaluate(real_df, real_df, synth_df, label_column="label")
    diag = eval_res.diagnostic_report
    assert diag is not None
    assert len(diag.broken_relationships) >= 1
    broken = diag.broken_relationships[0]
    assert (broken.col_a == "x" and broken.col_b == "y") or (broken.col_a == "y" and broken.col_b == "x")
    assert broken.delta > 0.5


def test_pareto_candidate_selection_prioritizes_privacy():
    """Verify that Pareto selection favors candidates satisfying privacy constraints."""
    t = Thresholds()
    # Candidate 1: High utility, but violates privacy (reid=0.03 > 0.01)
    c1 = CandidateEvaluation(
        candidate_index=0,
        plan=SynthesisConfig(plan_name="c1", recombination=0.1),
        utility=AuditMetrics(fidelity_score=0.98, utility_ratio=0.95, query_equivalence=0.98),
        attack=PrivacyAttackResults(re_id_rate=0.03, membership_accuracy=0.52),
        passed=False,
    )
    # Candidate 2: Moderate utility, satisfies privacy
    c2 = CandidateEvaluation(
        candidate_index=1,
        plan=SynthesisConfig(plan_name="c2", recombination=0.4),
        utility=AuditMetrics(fidelity_score=0.91, utility_ratio=0.88, query_equivalence=0.96),
        attack=PrivacyAttackResults(re_id_rate=0.005, membership_accuracy=0.53),
        passed=True,
    )

    best_idx = orchestrator.select_pareto_candidate([c1, c2], t)
    assert best_idx == 1
    assert orchestrator.decide(c2.utility, c2.attack, t)


def test_blackboard_state_tracking():
    """Verify that RunBlackboard tracks round results and diagnostic history."""
    bb = RunBlackboard(run_id="run_test_1")
    u = AuditMetrics(fidelity_score=0.9, utility_ratio=0.9, query_equivalence=0.95)
    a = PrivacyAttackResults(re_id_rate=0.005, membership_accuracy=0.52)
    plan = SynthesisConfig()

    rr = RoundResult(round_index=1, utility=u, attack=a, passed=True, plan=plan)
    bb.round_history.append(rr)
    bb.orchestrator_feedback_history.append("Great convergence")

    assert len(bb.round_history) == 1
    assert bb.orchestrator_feedback_history[0] == "Great convergence"
