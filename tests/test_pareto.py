"""Pareto frontier scores, skyline flags, and certified-candidate hand-off."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from agents import orchestrator
from main import BASE_DIR, export, set_active_candidate, ActiveCandidateRequest
from models import AuditMetrics, CandidateEvaluation, ForgeResponse, PrivacyAttackResults, SynthesisConfig, Thresholds
from pipeline import _PASSING_FRAMES, _RUN_CACHE, _RUN_STATES, activate_passing_candidate, get_passing_frame


def _eval(index, utility, reid, membership=0.52, attribute=0.0, linkage=0.0, near=0.0, passed=False):
    fidelity, tstr, query = utility
    return CandidateEvaluation(
        candidate_index=index,
        candidate_id=f"r1-c{index}",
        plan=SynthesisConfig(plan_name=f"c{index}", recombination=0.2),
        utility=AuditMetrics(fidelity_score=fidelity, utility_ratio=tstr, query_equivalence=query),
        attack=PrivacyAttackResults(
            re_id_rate=reid,
            membership_accuracy=membership,
            attribute_inference_advantage=attribute,
            linkage_rate=linkage,
            near_duplicate_rate=near,
        ),
        passed=passed,
        gate_passed=passed,
        utility_score=orchestrator.utility_score(
            AuditMetrics(fidelity_score=fidelity, utility_ratio=tstr, query_equivalence=query)
        ),
        privacy_risk=orchestrator.privacy_risk(
            PrivacyAttackResults(
                re_id_rate=reid,
                membership_accuracy=membership,
                attribute_inference_advantage=attribute,
                linkage_rate=linkage,
                near_duplicate_rate=near,
            ),
            Thresholds(),
        ),
    )


def test_utility_score_is_the_mean_of_the_three_gates():
    utility = AuditMetrics(fidelity_score=0.90, utility_ratio=0.84, query_equivalence=0.96)
    assert orchestrator.utility_score(utility) == pytest.approx((0.90 + 0.84 + 0.96) / 3.0)


def test_privacy_risk_is_the_worst_normalized_signal_and_ignores_near_duplicates():
    thresholds = Thresholds()
    quiet = PrivacyAttackResults(
        re_id_rate=0.005,
        membership_accuracy=0.55,
        attribute_inference_advantage=0.05,
        linkage_rate=0.03,
        near_duplicate_rate=1.0,
    )
    # 0.005/0.01 = 0.5; membership (0.55-0.5)/0.1 = 0.5; attribute 0.05/0.25 = 0.2; linkage 0.03/0.15 = 0.2
    assert orchestrator.privacy_risk(quiet, thresholds) == pytest.approx(0.5)

    leaked = PrivacyAttackResults(
        re_id_rate=0.03,
        membership_accuracy=0.50,
        attribute_inference_advantage=0.0,
        linkage_rate=0.0,
    )
    assert orchestrator.privacy_risk(leaked, thresholds) == pytest.approx(3.0)


def test_skyline_keeps_tradeoffs_and_drops_dominated_points():
    strong = _eval(0, (0.95, 0.95, 0.98), reid=0.002, membership=0.50, passed=True)
    private = _eval(1, (0.86, 0.86, 0.96), reid=0.001, membership=0.50, passed=True)
    dominated = _eval(2, (0.86, 0.86, 0.96), reid=0.004, membership=0.50, passed=True)
    failed = _eval(3, (0.99, 0.99, 0.99), reid=0.05, membership=0.50, passed=False)
    points = [strong, private, dominated, failed]
    orchestrator.mark_pareto(points)

    assert private.is_pareto_optimal
    assert strong.is_pareto_optimal
    assert failed.is_pareto_optimal
    assert not dominated.is_pareto_optimal
    assert dominated.gate_passed
    assert not failed.gate_passed


def test_winner_ranking_still_prefers_a_privacy_pass():
    thresholds = Thresholds()
    leaked = _eval(0, (0.98, 0.95, 0.98), reid=0.03, passed=False)
    safe = _eval(1, (0.91, 0.88, 0.96), reid=0.005, membership=0.53, passed=True)
    assert orchestrator.select_pareto_candidate([leaked, safe], thresholds) == 1


def test_only_a_passing_candidate_can_become_the_download():
    run_id = "pareto-hand-off"
    certified = pd.DataFrame({"age": [30, 40]})
    try:
        _RUN_STATES[run_id] = {
            "status": "SUCCESS",
            "progress": [],
            "response": ForgeResponse(
                run_id=run_id, status="SUCCESS", final_status="CERTIFIED", active_candidate_id="r1-c0",
            ).model_dump(),
        }
        _PASSING_FRAMES[run_id] = {"r1-c0": certified}
        _RUN_CACHE[run_id] = certified

        assert get_passing_frame(run_id, "r1-c1") is None
        assert activate_passing_candidate(run_id, "r9-c0") is None
        switched = set_active_candidate(run_id, ActiveCandidateRequest(candidate_id="r1-c0"))
        assert switched["rows"] == 2
        assert _RUN_STATES[run_id]["response"]["active_candidate_id"] == "r1-c0"

        downloaded = export(run_id, candidate_id="r1-c0")
        assert downloaded.filename == "synthetic_pareto-hand-off_r1-c0.csv"
        with pytest.raises(HTTPException) as missing:
            export(run_id, candidate_id="r2-c3")
        assert missing.value.status_code == 404

        _RUN_STATES[run_id]["response"]["status"] = "FAILED_TO_CONVERGE"
        with pytest.raises(HTTPException) as blocked:
            export(run_id, candidate_id="r1-c0")
        assert blocked.value.status_code == 403
        with pytest.raises(HTTPException) as blocked_switch:
            set_active_candidate(run_id, ActiveCandidateRequest(candidate_id="r1-c0"))
        assert blocked_switch.value.status_code == 403
    finally:
        _PASSING_FRAMES.pop(run_id, None)
        _RUN_CACHE.pop(run_id, None)
        _RUN_STATES.pop(run_id, None)
        for path in (
            BASE_DIR / "data" / "output" / f"{run_id}.csv",
            BASE_DIR / "data" / "output" / f"{run_id}_r1-c0.csv",
        ):
            Path(path).unlink(missing_ok=True)
