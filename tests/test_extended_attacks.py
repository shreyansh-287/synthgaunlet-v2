"""Extended attacks and the fairness audit.

An exact training copy fails the new attacks. Unrelated noise passes them.
"""

import numpy as np
import pandas as pd

from agents.attacks.registry import extended_attacks, run_attacks
from agents.fairness import audit
from agents.orchestrator import decide
from models import (
    AuditMetrics,
    ColumnProfile,
    DatasetProfile,
    PrivacyAttackResults,
    Thresholds,
)


def _frame(n=80, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "person": [f"p{i}" for i in range(n)],
        "gender": rng.choice(["F", "M"], n),
        "age": rng.randint(20, 70, n),
        "salary": np.arange(n) * 100,
    })


def _profile(frame: pd.DataFrame) -> DatasetProfile:
    sensitive = {"salary"}
    quasi = {"person", "gender", "age"}
    return DatasetProfile(
        columns=[
            ColumnProfile(
                name=column,
                dtype="categorical",
                stats={},
                is_quasi_identifier=column in quasi,
                is_sensitive=column in sensitive,
            )
            for column in frame.columns
        ],
        row_count=len(frame),
        label_column="salary",
    )


def _noise(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(1)
    noisy = frame.copy()
    noisy["person"] = [f"x{i}" for i in range(len(frame))]
    noisy["salary"] = rng.permutation(noisy["salary"].to_numpy())
    noisy["gender"] = rng.choice(["F", "M"], len(frame))
    noisy["age"] = rng.randint(20, 70, len(frame))
    return noisy


def test_exact_copy_fails_every_extended_attack_and_noise_passes():
    real = _frame()
    holdout = _frame(40, seed=2)
    profile = _profile(real)
    thresholds = Thresholds()
    copied = run_attacks(extended_attacks(), real, holdout, real.copy(), profile)
    noised = run_attacks(extended_attacks(), real, holdout, _noise(real), profile)

    assert copied["attribute_inference"].score > thresholds.attribute_inference_max
    assert copied["linkage"].score > thresholds.linkage_max
    assert copied["near_duplicate"].score > thresholds.near_duplicate_max
    assert noised["attribute_inference"].score <= thresholds.attribute_inference_max
    assert noised["linkage"].score <= thresholds.linkage_max
    assert noised["near_duplicate"].score <= thresholds.near_duplicate_max


def test_release_gate_rejects_the_new_attack_scores():
    utility = AuditMetrics(fidelity_score=0.95, utility_ratio=0.95, query_equivalence=0.99)
    leaking = PrivacyAttackResults(
        re_id_rate=0.0,
        membership_accuracy=0.5,
        linkage_rate=0.9,
    )
    assert decide(utility, leaking, Thresholds()) is False


def test_fairness_warns_on_a_representation_shift_and_passes_a_match():
    real = pd.DataFrame({
        "gender": ["F", "M"] * 40,
        "approved": [1, 0] * 40,
    })
    profile = DatasetProfile(
        columns=[
            ColumnProfile(name="gender", dtype="categorical", stats={}),
            ColumnProfile(name="approved", dtype="categorical", stats={}),
        ],
        row_count=len(real),
        label_column="approved",
    )
    skewed = real.copy()
    skewed["gender"] = ["F"] * 70 + ["M"] * 10
    warned = audit(profile, real, real, skewed)
    assert warned.status == "WARN"
    matched = audit(profile, real, real, real.copy())
    assert matched.status == "PASS"
