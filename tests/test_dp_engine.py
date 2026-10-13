"""Laplace marginal engine: budget, sensitivity, and the formal guarantee string."""

import numpy as np
import pandas as pd
import pytest

from agents.dp_engine import (
    NO_FORMAL_DP_GUARANTEE,
    BudgetExceeded,
    PrivacyBudgetAccountant,
    allocate_budget,
    apply_laplace,
    fit_marginals,
    formal_dp_guarantee_text,
    one_way_counts,
    sample,
    to_probabilities,
    two_way_counts,
)
from constants import DATASETS
from models import ForgeRequest, SourceSpec, Thresholds
from pipeline import run_forge


def test_accountant_rejects_spend_past_the_cap():
    accountant = PrivacyBudgetAccountant(1.0)
    accountant.spend("1-way:age", 0.4)
    accountant.spend("2-way:age|city", 0.6)
    assert accountant.spent == pytest.approx(1.0)
    assert accountant.remaining == pytest.approx(0.0)
    with pytest.raises(BudgetExceeded):
        accountant.spend("extra", 0.01)


def test_budget_shares_sum_to_the_total():
    shares = allocate_budget(1.0, 7)
    assert len(shares) == 7
    assert sum(shares) == pytest.approx(1.0)
    assert all(share > 0 for share in shares)
    weighted = allocate_budget(1.0, 3, [1.0, 1.0, 3.0])
    assert sum(weighted) == pytest.approx(1.0)
    assert weighted[2] > weighted[0]


def test_laplace_clips_negative_counts_and_renormalizes():
    rng = np.random.RandomState(0)
    noisy = apply_laplace(np.array([1.0, 1.0, 1.0]), epsilon=0.01, rng=rng)
    assert noisy.min() >= 0
    probs = to_probabilities(noisy)
    assert probs.sum() == pytest.approx(1.0)


def test_histogram_counts_have_sensitivity_one():
    cities = ["a", "a", "b", "c"]
    levels = ["a", "b", "c"]
    base = one_way_counts(pd.Series(cities), levels)
    added = one_way_counts(pd.Series(cities + ["b"]), levels)
    assert np.abs(added - base).sum() == 1

    left = pd.Series(["a", "a", "b"])
    right = pd.Series(["x", "y", "x"])
    base_joint = two_way_counts(left, right, ["a", "b"], ["x", "y"])
    extra_joint = two_way_counts(
        pd.concat([left, pd.Series(["b"])], ignore_index=True),
        pd.concat([right, pd.Series(["y"])], ignore_index=True),
        ["a", "b"],
        ["x", "y"],
    )
    assert np.abs(extra_joint - base_joint).sum() == 1


def test_noisy_marginals_spend_at_most_the_budget_and_preserve_dependence():
    age = np.repeat([20, 60], 200)
    frame = pd.DataFrame({
        "age": age,
        "city": np.where(age == 20, "young", "older"),
        "label": np.where(age == 20, 0, 1),
    })
    model = fit_marginals(frame, epsilon_total=8.0, label_column="label", seed=3)
    assert model.epsilon_spent <= model.epsilon_total + 1e-9
    assert model.epsilon_spent == pytest.approx(8.0)
    assert any(name.startswith("2-way:") for name in model.query_names)
    drawn = sample(model, 300, seed=4)
    assert list(drawn.columns) == ["age", "city", "label"]
    assert len(drawn) == 300
    agreement = (drawn["city"].eq("young") == drawn["age"].eq(20)).mean()
    assert agreement > 0.8


def test_numeric_draws_spread_inside_the_released_bins():
    rng = np.random.RandomState(0)
    frame = pd.DataFrame({
        "amount": rng.normal(loc=50, scale=10, size=500),
        "label": rng.randint(0, 2, size=500),
    })
    model = fit_marginals(frame, epsilon_total=2.0, label_column="label", seed=1)
    drawn = sample(model, 300, seed=2)
    assert drawn["amount"].nunique() > 8
    assert float(drawn["amount"].min()) >= float(frame["amount"].min()) - 1e-6
    assert float(drawn["amount"].max()) <= float(frame["amount"].max()) + 1e-6


def test_label_keeps_a_two_way_when_other_columns_move_together():
    rng = np.random.RandomState(0)
    shared = rng.normal(size=400)
    frame = pd.DataFrame({
        "a": shared,
        "b": shared + rng.normal(scale=0.01, size=400),
        "c": shared + rng.normal(scale=0.01, size=400),
        "d": shared + rng.normal(scale=0.01, size=400),
        "label": (shared > 0).astype(int),
    })
    model = fit_marginals(frame, epsilon_total=1.0, label_column="label", seed=1)
    assert model.epsilon_spent == pytest.approx(1.0)
    assert any(name.startswith("2-way:") and "label" in name for name in model.query_names)


def test_guarantee_text_is_formal_only_for_a_pure_laplace_release():
    assert formal_dp_guarantee_text(False, 1.0) == NO_FORMAL_DP_GUARANTEE
    assert formal_dp_guarantee_text(True, 1.0) == "Formal DP guarantee: epsilon = 1 (Laplace mechanism)"
    assert "Laplace mechanism" in formal_dp_guarantee_text(True, 0.5)


def test_pure_dp_run_reports_the_spent_epsilon():
    if not DATASETS["health_diabetes"]["path"].exists():
        pytest.skip("health_diabetes CSV missing")
    events = []
    response = run_forge(
        ForgeRequest(
            source=SourceSpec(type="csv", dataset_key="health_diabetes"),
            n_rows=80,
            thresholds=Thresholds(max_rounds=1),
            auto_heal_enabled=False,
            synthesis_mode="dp",
            dp_epsilon=1.0,
        ),
        progress=events.append,
    )
    assert any(event["type"] == "dp_budget" for event in events)
    assert response.formal_dp_guarantee.startswith("Formal DP guarantee: epsilon =")
    assert response.formal_dp_guarantee.endswith("(Laplace mechanism)")
    assert "No formal DP guarantee" not in response.formal_dp_guarantee
    assert "DIFFERENTIAL PRIVACY:" in response.scorecard_summary
