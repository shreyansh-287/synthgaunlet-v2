"""Auxiliary linkage: unique public keys that point at exactly one synthetic row."""

from __future__ import annotations

from typing import List

import pandas as pd

from agents.attacks.base import BaseAttack, PluginAttackResult
from agents.profiler import is_numeric
from models import DatasetProfile

_HINTS = ("zip", "postal", "gender", "sex", "age", "birth", "year", "race", "marital")


def _public_columns(profile: DatasetProfile, frame: pd.DataFrame) -> List[str]:
    quasi = [
        column.name
        for column in profile.columns
        if column.is_quasi_identifier and column.name in frame.columns
    ]
    preferred = [name for name in quasi if any(hint in name.lower() for hint in _HINTS)]
    return (preferred or quasi)[:3]


def _keys(frame: pd.DataFrame, columns: List[str]) -> pd.Series:
    parts = []
    for name in columns:
        column = frame[name]
        if is_numeric(column) and "age" in name.lower():
            numeric = pd.to_numeric(column, errors="coerce").fillna(-1)
            parts.append((numeric // 5 * 5).astype(int).astype(str))
        else:
            parts.append(column.fillna("<NA>").astype(str).str.strip())
    keys = parts[0]
    for part in parts[1:]:
        keys = keys.str.cat(part, sep="|", na_rep="<NA>")
    return keys


class LinkageAttack(BaseAttack):
    name = "linkage"

    def run(self, real_train, real_holdout, synth, profile) -> PluginAttackResult:
        del real_holdout
        columns = [name for name in _public_columns(profile, real_train) if name in synth.columns]
        if not columns:
            return PluginAttackResult(self.name, 0.0, True, "No public quasi-identifiers to link.")
        real_keys = _keys(real_train, columns)
        synth_keys = _keys(synth, columns)
        real_counts = real_keys.value_counts()
        synth_counts = synth_keys.value_counts()
        unique_real = real_counts[real_counts == 1].index
        if len(unique_real) == 0:
            return PluginAttackResult(self.name, 0.0, True, "No unique public keys in the training table.")
        linked = sum(1 for key in unique_real if int(synth_counts.get(key, 0)) == 1)
        rate = float(linked / len(unique_real))
        return PluginAttackResult(
            self.name,
            rate,
            True,
            f"{linked} of {len(unique_real)} unique public keys link to one synthetic row.",
            {"n_unique": int(len(unique_real)), "n_linked": int(linked), "columns": columns},
        )
