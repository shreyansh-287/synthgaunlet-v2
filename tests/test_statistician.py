"""Utility-metric tests: the metric must discriminate useful vs useless data."""

import numpy as np
import pandas as pd

from agents import statistician


def _dataset(n=600, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.3, n) > 0).astype(int).astype(str)
    return pd.DataFrame({"feat1": x, "feat2": rng.normal(5, 2, n), "label": y})


def test_high_fidelity_for_similar_data():
    real = _dataset()
    synth = _dataset(seed=1)  # same generative process
    fidelity, _ = statistician.distribution_fidelity(real, synth)
    assert fidelity > 0.8


def test_tstr_high_for_faithful_synth_low_for_noise():
    real = _dataset(800)
    train, test = real.iloc[:560], real.iloc[560:]

    good_synth = _dataset(560, seed=7)  # captures the real signal
    good = statistician.tstr(train, test, good_synth, "label")

    rng = np.random.RandomState(3)
    noise = pd.DataFrame(
        {
            "feat1": rng.normal(0, 1, 560),
            "feat2": rng.normal(5, 2, 560),
            "label": rng.choice(["0", "1"], 560),  # label unrelated to features
        }
    )
    bad = statistician.tstr(train, test, noise, "label")

    assert good["utility_ratio"] > bad["utility_ratio"]
