"""Near-duplicate check: synthetic rows that sit too close to a training record."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from agents.attacks.base import BaseAttack, PluginAttackResult
from agents.profiler import is_numeric

# Standardized Euclidean distance. An exact copy is 0. A single categorical
# mismatch is about 1.4, so 0.5 flags only near-copies.
NEAR_DUPLICATE_DELTA = 0.5


def _encode(real: pd.DataFrame, synth: pd.DataFrame, columns: list):
    numeric = [column for column in columns if is_numeric(real[column])]
    categorical = [column for column in columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    encoder = ColumnTransformer(transformers, remainder="drop")
    union = pd.concat([real[columns], synth[columns]], ignore_index=True)
    for column in categorical:
        union[column] = union[column].fillna("missing").astype(str)
        real[column] = real[column].fillna("missing").astype(str)
        synth[column] = synth[column].fillna("missing").astype(str)
    encoder.fit(union)
    return encoder.transform(real[columns]), encoder.transform(synth[columns])


class NearDuplicateAttack(BaseAttack):
    name = "near_duplicate"

    def run(self, real_train, real_holdout, synth, profile) -> PluginAttackResult:
        del real_holdout
        # Whole-row distance. Quasi-identifiers alone are often shared by many
        # different people, so a demographic match is not a duplicate record.
        columns = [name for name in real_train.columns if name in synth.columns]
        if not columns:
            return PluginAttackResult(self.name, 0.0, True, "No quasi-identifiers for the near-duplicate check.")
        real_view = real_train[columns].copy()
        synth_view = synth[columns].copy()
        real_enc, synth_enc = _encode(real_view, synth_view, columns)
        if getattr(real_enc, "shape", (0, 0))[1] == 0:
            return PluginAttackResult(self.name, 0.0, True, "Quasi-identifiers could not be encoded.")
        neighbors = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(real_enc)
        distances, _indices = neighbors.kneighbors(synth_enc)
        normalized = distances.ravel()
        rate = float((normalized < NEAR_DUPLICATE_DELTA).mean())
        return PluginAttackResult(
            self.name,
            rate,
            True,
            f"{rate:.1%} of synthetic rows are within {NEAR_DUPLICATE_DELTA} of a training record.",
            {"delta": NEAR_DUPLICATE_DELTA, "n_close": int((normalized < NEAR_DUPLICATE_DELTA).sum())},
        )
