"""Auto-Heal rescue ladder: knobs, row repairs, and the pipeline gate."""

import pandas as pd
import pytest

from agents.generator import drop_low_dcr_rows, generalize_frame
from agents.orchestrator import build_rescue_plan
from constants import DATASETS
from models import ForgeRequest, SourceSpec, SynthesisConfig, Thresholds
from pipeline import _certification, run_forge


def _plan() -> SynthesisConfig:
    return SynthesisConfig(
        plan_name="balanced",
        recombination=0.2,
        numeric_noise=0.1,
        categorical_shuffle=0.05,
        rare_category_floor=0.01,
    )


def test_tighten_plan_raises_the_three_privacy_knobs():
    plan = build_rescue_plan(1, _plan(), ["age"])
    assert plan.recombination >= 0.65
    assert plan.numeric_noise == pytest.approx(0.15)
    assert plan.rare_category_floor == 0.05
    assert plan.plan_name == "rescue_tighten"


def test_second_generation_plan_increases_perturbation():
    plan = build_rescue_plan(3, _plan(), ["age"])
    assert plan.recombination >= 0.85
    assert plan.numeric_noise > 0.1
    assert plan.copula_rank_preservation is False


def test_drop_low_dcr_rows_removes_exact_copies():
    real = pd.DataFrame({"age": list(range(30, 50)), "city": ["a"] * 20})
    close = real.iloc[:10].copy()
    far = pd.DataFrame({"age": list(range(80, 90)), "city": ["z"] * 10})
    synth = pd.concat([close, far], ignore_index=True)
    kept = drop_low_dcr_rows(synth, real, ["age", "city"])
    assert len(kept) < len(synth)
    assert kept["age"].min() >= 80


def test_generalize_buckets_age_collapses_rare_categories_and_winsorizes():
    real = pd.DataFrame({
        "age": list(range(20, 80)),
        "income": [10.0] * 30 + [1000.0] * 30,
        "city": ["common"] * 59 + ["rare"] * 1,
        "outcome": [0, 1] * 30,
    })
    frame = real.copy()
    frame.loc[0, "income"] = -500.0
    frame.loc[1, "income"] = 5000.0
    out = generalize_frame(frame, real, ["age", "city"], label_column="outcome")
    assert set(out["age"].dropna().unique()).issubset(set(range(0, 85, 5)))
    assert "rare" not in set(out["city"].astype(str))
    assert "Other" in set(out["city"].astype(str))
    assert out["income"].min() > -500
    assert out["income"].max() < 5000
    assert set(out["outcome"]) == {0, 1}


def test_certification_names():
    assert _certification(True, False) == "CERTIFIED"
    assert _certification(True, True) == "CERTIFIED_AFTER_RESCUE"
    assert _certification(False, False) == "NOT_CERTIFIED"


def _harsh() -> Thresholds:
    return Thresholds(
        max_rounds=1,
        fidelity_min=1.0,
        tstr_min=1.0,
        query_equivalence_min=1.0,
        reid_max=0.0,
        membership_max=0.0,
    )


def test_failed_run_streams_the_rescue_ladder_and_stops_on_pass():
    if not DATASETS["health_diabetes"]["path"].exists():
        return
    events = []
    response = run_forge(
        ForgeRequest(
            source=SourceSpec(type="csv", dataset_key="health_diabetes"),
            n_rows=120,
            thresholds=_harsh(),
            auto_heal_enabled=True,
        ),
        progress=events.append,
    )
    kinds = [event["type"] for event in events]
    assert "rescue_started" in kinds
    assert "rescue_finished" in kinds
    assert response.rescue_attempts
    assert response.rescue_attempts[0].name == "tighten"
    assert response.final_status in {"CERTIFIED_AFTER_RESCUE", "NOT_CERTIFIED"}
    if response.final_status == "CERTIFIED_AFTER_RESCUE":
        assert response.status == "SUCCESS"
        assert response.rescue_attempts[-1].passed or any(item.passed for item in response.rescue_attempts)
        assert len(response.rescue_attempts) < 3 or response.rescue_attempts[-1].passed
    else:
        assert response.status == "FAILED_TO_CONVERGE"
        assert len(response.rescue_attempts) == 3
        assert all(not item.passed for item in response.rescue_attempts)


def test_auto_heal_can_be_turned_off():
    if not DATASETS["health_diabetes"]["path"].exists():
        return
    events = []
    response = run_forge(
        ForgeRequest(
            source=SourceSpec(type="csv", dataset_key="health_diabetes"),
            n_rows=120,
            thresholds=_harsh(),
            auto_heal_enabled=False,
        ),
        progress=events.append,
    )
    assert "rescue_started" not in [event["type"] for event in events]
    assert response.rescue_attempts == []
    assert response.status == "FAILED_TO_CONVERGE"
    assert response.final_status == "NOT_CERTIFIED"
