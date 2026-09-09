"""Attack-tool sanity tests: the attacks must actually detect leakage."""

import numpy as np
import pandas as pd

from agents import attack_tools


def _toy(n=300, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "age": rng.randint(18, 80, n),
            "zip": rng.choice(["90210", "10001", "60601", "94103"], n),
            "gender": rng.choice(["M", "F"], n),
            "income": rng.normal(50000, 15000, n).round(),
        }
    )


def test_identical_copy_reidentifies_the_rare_individuals():
    """An exact copy re-identifies every genuinely rare (non k-anonymous)
    person -- but not those hiding in a crowd of identical quasi-identifiers.
    So the re-id rate must equal the fraction of rare records and be high."""
    real = _toy()
    synth = real.copy()
    cols = ["age", "zip", "gender"]
    out = attack_tools.run_reidentification(real, synth, cols, threshold=0.1)

    keys = attack_tools._kanon_keys(real, cols)
    rare_frac = float((keys.map(keys.value_counts()) < 5).mean())

    assert out["re_id_rate"] > 0.3
    assert abs(out["re_id_rate"] - rare_frac) < 0.05


def test_heavy_noise_reduces_reidentification():
    real = _toy()
    synth = real.copy()
    synth["age"] = synth["age"] + np.random.RandomState(1).randint(-25, 25, len(synth))
    synth["zip"] = np.random.RandomState(2).choice(
        ["90210", "10001", "60601", "94103"], len(synth)
    )
    noised = attack_tools.run_reidentification(real, synth, ["age", "zip", "gender"], threshold=0.05)
    identical = attack_tools.run_reidentification(real, real.copy(), ["age", "zip", "gender"], threshold=0.05)
    assert noised["re_id_rate"] < identical["re_id_rate"]


def test_membership_inference_returns_probability():
    real = _toy(400)
    real_in = real.iloc[:200]
    real_out = real.iloc[200:]
    synth = real_in.copy()  # synth built from IN -> membership signal exists
    out = attack_tools.run_membership_inference(
        real_in, real_out, synth, ["age", "zip", "gender", "income"]
    )
    assert 0.0 <= out["membership_accuracy"] <= 1.0


def test_reidentification_handles_nan_quasi_identifiers():
    """BigQuery tables often have missing/null values in quasi-identifiers.
    The re-identification calculation must never fail with KeyError: nan."""
    real = pd.DataFrame({
        "user_name": ["alice", np.nan, "charlie", "alice", None],
        "artist_name": ["coldplay", "beatles", np.nan, "coldplay", "queen"],
        "artist_mbids": [np.nan, "mbid-1", "mbid-2", np.nan, np.nan],
        "age": [25.0, np.nan, 42.0, 25.0, 60.0]
    })
    synth = real.copy()
    out = attack_tools.run_reidentification(
        real, synth, ["user_name", "artist_name", "artist_mbids", "age"], threshold=0.1
    )
    assert "re_id_rate" in out
    assert 0.0 <= out["re_id_rate"] <= 1.0


def test_outlier_linkage_handles_nan_and_single_column():
    """Outlier linkage must handle single-column QIs with nulls without TypeError."""
    real = pd.DataFrame({"country_code": [np.nan, "US", "IN", float("nan"), None, "FR", "DE"]})
    synth = pd.DataFrame({"country_code": ["US", "IN", "GB", "FR"]})
    out = attack_tools.run_outlier_linkage(real, synth, ["country_code"])
    assert "outlier_linkage_rate" in out
    assert 0.0 <= out["outlier_linkage_rate"] <= 1.0


