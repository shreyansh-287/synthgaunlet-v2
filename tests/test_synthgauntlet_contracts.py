import pandas as pd

from agents import orchestrator
from agents.profiler import profile_dataframe
from models import AuditMetrics, PrivacyAttackResults, RoundResult, SynthesisConfig, Thresholds
from reporter import build_dual_scorecard


def test_release_gate_requires_all_hard_thresholds():
    passing_utility = AuditMetrics(
        fidelity_score=0.95,
        query_equivalence=0.97,
        utility_ratio=0.94,
    )
    passing_privacy = PrivacyAttackResults(
        re_id_rate=0.005,
        membership_accuracy=0.52,
    )
    thresholds = Thresholds()
    assert orchestrator.decide(passing_utility, passing_privacy, thresholds)

    failing_utility = passing_utility.model_copy(update={"query_equivalence": 0.94})
    assert not orchestrator.decide(failing_utility, passing_privacy, thresholds)


def test_successful_round_produces_dual_scorecard_contract():
    utility = AuditMetrics(
        fidelity_score=0.96,
        query_equivalence=0.98,
        utility_ratio=0.95,
    )
    privacy = PrivacyAttackResults(
        re_id_rate=0.004,
        membership_accuracy=0.51,
    )
    round_result = RoundResult(
        round_index=1,
        utility=utility,
        attack=privacy,
        passed=True,
        plan=SynthesisConfig(recombination=0.4),
    )
    scorecard = build_dual_scorecard([round_result], "SUCCESS", Thresholds())
    assert scorecard.passed
    assert scorecard.utility.utility_ratio >= scorecard.thresholds.tstr_min
    assert scorecard.privacy.re_id_rate <= scorecard.thresholds.reid_max


def test_profiler_emits_aggregates_and_rare_combination_counts():
    frame = pd.DataFrame({"age": [20, 21, 21], "zip": ["a", "b", "b"], "outcome": [0, 1, 1]})
    profile = profile_dataframe(frame, quasi_identifiers=["age", "zip"], label_column="outcome")
    assert profile.row_count == 3
    assert profile.missingness["age"] == 0.0
    assert profile.rare_combinations
    assert "values" not in profile.rare_combinations[0]
