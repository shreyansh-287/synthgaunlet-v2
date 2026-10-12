"""Attribute inference: predict a sensitive column from quasi-identifiers."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from agents.attack_tools import _encode_pair
from agents.attacks.base import BaseAttack, PluginAttackResult
from agents.profiler import is_numeric
from models import DatasetProfile

_SAMPLE = 1200


def _sensitive_column(profile: DatasetProfile, frame: pd.DataFrame) -> Optional[str]:
    for column in profile.columns:
        if column.is_sensitive and column.name in frame.columns:
            return column.name
    return None


def _quasi_identifiers(profile: DatasetProfile, frame: pd.DataFrame, sensitive: str) -> List[str]:
    names = [
        column.name
        for column in profile.columns
        if column.is_quasi_identifier and column.name in frame.columns and column.name != sensitive
    ]
    return names


def _target_codes(real: pd.Series, synth: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encode the sensitive column in one shared code space."""
    if is_numeric(real) and real.nunique(dropna=True) > 8:
        real_num = pd.to_numeric(real, errors="coerce")
        fill = float(real_num.median()) if real_num.notna().any() else 0.0
        real_num = real_num.fillna(fill)
        synth_num = pd.to_numeric(synth, errors="coerce").fillna(fill)
        edges = np.unique(np.quantile(real_num.to_numpy(dtype=float), [0, 0.25, 0.5, 0.75, 1.0]))
        if len(edges) >= 3:
            real_bins = np.digitize(real_num.to_numpy(dtype=float), edges[1:-1], right=False)
            synth_bins = np.digitize(synth_num.to_numpy(dtype=float), edges[1:-1], right=False)
            return real_bins.astype(int), synth_bins.astype(int)
    classes = list(dict.fromkeys(real.fillna("missing").astype(str).tolist()))
    mapping = {label: index for index, label in enumerate(classes)}
    fallback = 0
    y_real = real.fillna("missing").astype(str).map(mapping).to_numpy(dtype=int)
    y_synth = synth.fillna("missing").astype(str).map(lambda value: mapping.get(value, fallback)).to_numpy(dtype=int)
    return y_real, y_synth


class AttributeInferenceAttack(BaseAttack):
    name = "attribute_inference"

    def run(self, real_train, real_holdout, synth, profile) -> PluginAttackResult:
        del real_holdout  # The lookup is against people the generator was trained on.
        sensitive = _sensitive_column(profile, real_train)
        if sensitive is None or sensitive not in synth.columns:
            return PluginAttackResult(self.name, 0.0, True, "No sensitive column to infer.")
        features = _quasi_identifiers(profile, real_train, sensitive)
        features = [column for column in features if column in synth.columns]
        if not features:
            return PluginAttackResult(self.name, 0.0, True, "No quasi-identifiers for attribute inference.")

        train = real_train if len(real_train) <= _SAMPLE else real_train.sample(_SAMPLE, random_state=42)
        synth_fit = synth if len(synth) <= _SAMPLE else synth.sample(_SAMPLE, random_state=42)
        real_enc, synth_enc, _used = _encode_pair(train, synth_fit, features)
        if real_enc.shape[1] == 0:
            return PluginAttackResult(self.name, 0.0, True, "Quasi-identifiers could not be encoded.")

        y_real, y_synth = _target_codes(train[sensitive], synth_fit[sensitive])
        if len(np.unique(y_synth)) < 2 or len(np.unique(y_real)) < 2:
            return PluginAttackResult(self.name, 0.0, True, "Sensitive column has a single value.")

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(synth_enc, y_synth)
        predicted = model.predict(real_enc)
        accuracy = float((predicted == y_real).mean())
        majority = float(np.bincount(y_real).max() / len(y_real))
        advantage = float(max(0.0, accuracy - majority))
        return PluginAttackResult(
            self.name,
            advantage,
            True,
            f"kNN advantage {advantage:.3f} over a {majority:.3f} majority baseline.",
            {"accuracy": accuracy, "baseline": majority, "sensitive": sensitive, "features": features},
        )
