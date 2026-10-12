"""Laplace noisy-marginal synthesizer.

Counting queries have sensitivity 1: adding or removing one person changes a
single histogram cell by 1. Noise scale is therefore 1 / epsilon_i. Quantile
edges and category domains are the histogram grid. Sampling more rows from the
released noisy probabilities is post-processing and does not spend more budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models import SynthesisConfig

NO_FORMAL_DP_GUARANTEE = "No formal DP guarantee"
_PAIR_CORR_MIN = 0.2
_MAX_PAIRS = 4
_MAX_CATEGORIES = 20
_QUANTILE_BINS = 5
_LOW_CARDINALITY = 12


class BudgetExceeded(ValueError):
    """Raised when a query would push the spent epsilon above the cap."""


class PrivacyBudgetAccountant:
    """Enforces sum of per-query epsilons <= epsilon_total."""

    def __init__(self, epsilon_total: float):
        total = float(epsilon_total)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("epsilon_total must be a positive finite number.")
        self.epsilon_total = total
        self._spent = 0.0
        self.entries: List[Tuple[str, float]] = []

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self.epsilon_total - self._spent

    def spend(self, name: str, epsilon: float) -> float:
        amount = float(epsilon)
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError(f"Epsilon for {name} must be positive.")
        if self._spent + amount > self.epsilon_total + 1e-9:
            raise BudgetExceeded(
                f"Query {name} needs epsilon {amount}, but only {self.remaining} remains "
                f"of {self.epsilon_total}."
            )
        self._spent += amount
        self.entries.append((name, amount))
        return amount


def allocate_budget(epsilon_total: float, n_queries: int) -> List[float]:
    """Split a budget into n positive shares that sum to epsilon_total."""
    if n_queries < 1:
        raise ValueError("At least one query is required.")
    total = float(epsilon_total)
    raw = np.full(n_queries, total / n_queries, dtype=float)
    raw[-1] = total - float(raw[:-1].sum())
    if raw[-1] <= 0:
        raw = np.full(n_queries, total / n_queries, dtype=float)
        raw *= total / float(raw.sum())
    return [float(part) for part in raw]


def apply_laplace(counts: np.ndarray, epsilon: float, rng: np.random.RandomState) -> np.ndarray:
    """Laplace mechanism, sensitivity 1. Negative noisy counts are clipped to 0."""
    scale = 1.0 / float(epsilon)
    noisy = np.asarray(counts, dtype=float) + rng.laplace(0.0, scale, size=np.shape(counts))
    return np.clip(noisy, 0.0, None)


def to_probabilities(noisy_counts: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(noisy_counts, dtype=float), 0.0, None)
    total = float(values.sum())
    if total <= 0:
        size = int(values.size) or 1
        return np.full(size, 1.0 / size, dtype=float).reshape(values.shape if values.size else (1,))
    return values / total


def formal_dp_guarantee_text(pure: bool, epsilon: float) -> str:
    if not pure:
        return NO_FORMAL_DP_GUARANTEE
    text = f"{float(epsilon):.4f}".rstrip("0").rstrip(".")
    return f"Formal DP guarantee: epsilon = {text} (Laplace mechanism)"


@dataclass
class DPMarginals:
    columns: List[str]
    levels: Dict[str, list]
    one_way: Dict[str, np.ndarray]
    two_way: Dict[Tuple[str, str], np.ndarray]
    epsilon_total: float
    epsilon_spent: float
    query_names: List[str] = field(default_factory=list)
    label_column: Optional[str] = None


def release_plan(model: DPMarginals) -> SynthesisConfig:
    return SynthesisConfig(
        plan_name="dp_laplace",
        recombination=1.0,
        numeric_noise=0.0,
        categorical_shuffle=1.0,
        copula_rank_preservation=False,
        rationale=(
            "Laplace mechanism on 1-way marginals and high-correlation 2-way tables. "
            f"Epsilon spent {model.epsilon_spent}."
        ),
    )


def one_way_counts(values: pd.Series, levels: list) -> np.ndarray:
    index = {level: i for i, level in enumerate(levels)}
    counts = np.zeros(len(levels), dtype=float)
    for value in values.tolist():
        slot = index.get(value)
        if slot is not None:
            counts[slot] += 1.0
    return counts


def two_way_counts(left: pd.Series, right: pd.Series, left_levels: list, right_levels: list) -> np.ndarray:
    left_index = {level: i for i, level in enumerate(left_levels)}
    right_index = {level: i for i, level in enumerate(right_levels)}
    counts = np.zeros((len(left_levels), len(right_levels)), dtype=float)
    for a, b in zip(left.tolist(), right.tolist()):
        i = left_index.get(a)
        j = right_index.get(b)
        if i is not None and j is not None:
            counts[i, j] += 1.0
    return counts


def _is_continuous(series: pd.Series) -> bool:
    numeric = pd.api.types.is_numeric_dtype(series)
    if not numeric:
        return False
    return int(series.nunique(dropna=True)) > _LOW_CARDINALITY


def _fill_missing(series: pd.Series) -> pd.Series:
    if not series.isna().any():
        return series
    mode = series.mode(dropna=True)
    fill_value = mode.iloc[0] if not mode.empty else "missing"
    return series.fillna(fill_value)


def _category_levels(series: pd.Series, protect: bool) -> list:
    counts = series.astype(str).value_counts()
    levels = list(counts.index)
    if protect or len(levels) <= _MAX_CATEGORIES:
        return levels
    return levels[: _MAX_CATEGORIES - 1] + ["Other"]


def _project_categories(series: pd.Series, levels: list) -> pd.Series:
    allowed = set(levels)
    as_text = series.astype(str)
    if "Other" in allowed:
        return as_text.where(as_text.isin(allowed), "Other")
    return as_text.where(as_text.isin(allowed), levels[0])


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)).astype(float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def _discretize(frame: pd.DataFrame, label_column: Optional[str]) -> tuple[pd.DataFrame, Dict[str, list]]:
    coded = pd.DataFrame(index=frame.index)
    levels: Dict[str, list] = {}
    for column in frame.columns:
        series = frame[column]
        if _is_continuous(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            numeric = numeric.fillna(fill).to_numpy(dtype=float)
            unique = np.unique(numeric)
            bins = min(_QUANTILE_BINS, max(1, len(unique)))
            if bins == 1:
                levels[column] = [float(unique[0])]
                coded[column] = 0
                continue
            edges = _quantile_edges(numeric, bins)
            index = np.digitize(numeric, edges[1:-1], right=False)
            index = np.clip(index, 0, bins - 1)
            integer_source = pd.api.types.is_integer_dtype(series)
            mids = []
            for left, right in zip(edges[:-1], edges[1:]):
                mid = (left + right) / 2.0
                mids.append(int(round(mid)) if integer_source else float(mid))
            levels[column] = mids
            coded[column] = index.astype(int)
            continue
        filled = _fill_missing(series)
        if pd.api.types.is_numeric_dtype(filled):
            column_levels = sorted(pd.unique(filled).tolist())
            mapping = {level: i for i, level in enumerate(column_levels)}
            levels[column] = column_levels
            coded[column] = filled.map(mapping).astype(int)
            continue
        column_levels = _category_levels(filled, protect=column == label_column)
        projected = _project_categories(filled, column_levels)
        mapping = {level: i for i, level in enumerate(column_levels)}
        levels[column] = column_levels
        coded[column] = projected.map(mapping).astype(int)
    return coded, levels


def _high_correlation_pairs(coded: pd.DataFrame) -> List[Tuple[str, str, float]]:
    if coded.shape[1] < 2:
        return []
    corr = coded.corr(numeric_only=True).fillna(0.0).abs()
    pairs: List[Tuple[str, str, float]] = []
    columns = list(coded.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            strength = float(corr.loc[left, right])
            if np.isfinite(strength):
                pairs.append((left, right, strength))
    pairs.sort(key=lambda item: item[2], reverse=True)
    chosen = [item for item in pairs if item[2] >= _PAIR_CORR_MIN][:_MAX_PAIRS]
    if not chosen and pairs and pairs[0][2] > 0:
        chosen = pairs[:1]
    return chosen


def fit_marginals(
    frame: pd.DataFrame,
    epsilon_total: float,
    label_column: Optional[str] = None,
    seed: int = 0,
) -> DPMarginals:
    """Release noisy 1-way marginals and high-correlation 2-way tables."""
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("Differential privacy synthesis needs a non-empty table.")
    coded, levels = _discretize(frame, label_column)
    pairs = _high_correlation_pairs(coded)
    query_names = [f"1-way:{column}" for column in coded.columns]
    query_names.extend(f"2-way:{left}|{right}" for left, right, _strength in pairs)
    shares = allocate_budget(epsilon_total, len(query_names))
    accountant = PrivacyBudgetAccountant(epsilon_total)
    rng = np.random.RandomState(seed)

    one_way: Dict[str, np.ndarray] = {}
    for column, share in zip(coded.columns, shares[: coded.shape[1]]):
        accountant.spend(f"1-way:{column}", share)
        counts = np.bincount(coded[column].to_numpy(), minlength=len(levels[column])).astype(float)
        one_way[column] = to_probabilities(apply_laplace(counts, share, rng)).reshape(-1)

    two_way: Dict[Tuple[str, str], np.ndarray] = {}
    for (left, right, _strength), share in zip(pairs, shares[coded.shape[1] :]):
        accountant.spend(f"2-way:{left}|{right}", share)
        counts = np.zeros((len(levels[left]), len(levels[right])), dtype=float)
        np.add.at(counts, (coded[left].to_numpy(), coded[right].to_numpy()), 1.0)
        two_way[(left, right)] = to_probabilities(apply_laplace(counts, share, rng))

    return DPMarginals(
        columns=list(frame.columns),
        levels=levels,
        one_way=one_way,
        two_way=two_way,
        epsilon_total=float(epsilon_total),
        epsilon_spent=accountant.spent,
        query_names=query_names,
        label_column=label_column if label_column in frame.columns else None,
    )


def _joint(model: DPMarginals, parent: str, child: str) -> Optional[np.ndarray]:
    direct = model.two_way.get((parent, child))
    if direct is not None:
        return direct
    flipped = model.two_way.get((child, parent))
    if flipped is not None:
        return flipped.T
    return None


def _best_parent(model: DPMarginals, column: str, assigned: Dict[str, np.ndarray]) -> Optional[str]:
    best_name = None
    best_mass = -1.0
    for parent in assigned:
        joint = _joint(model, parent, column)
        if joint is None:
            continue
        # Prefer the parent whose noisy table is least uniform, a stand-in for strength.
        flat = joint.ravel()
        peak = float(flat.max()) if flat.size else 0.0
        if peak > best_mass:
            best_mass = peak
            best_name = parent
    return best_name


def sample(model: DPMarginals, n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Draw rows from the released noisy conditionals. This spends no more epsilon."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive.")
    rng = np.random.RandomState(seed)
    order = list(model.columns)
    if model.label_column in order:
        order.remove(model.label_column)
        order.insert(0, model.label_column)

    assigned: Dict[str, np.ndarray] = {}
    for column in order:
        parent = _best_parent(model, column, assigned)
        joint = _joint(model, parent, column) if parent else None
        if parent is None or joint is None:
            assigned[column] = rng.choice(len(model.one_way[column]), size=n_rows, p=to_probabilities(model.one_way[column]))
            continue
        parent_idx = assigned[parent]
        drawn = np.empty(n_rows, dtype=int)
        for level in range(joint.shape[0]):
            mask = parent_idx == level
            count = int(mask.sum())
            if count == 0:
                continue
            cond = joint[level]
            probs = to_probabilities(cond if float(cond.sum()) > 0 else model.one_way[column])
            drawn[mask] = rng.choice(len(probs), size=count, p=probs)
        assigned[column] = drawn

    out = {}
    for column in model.columns:
        values = model.levels[column]
        out[column] = [values[int(i)] for i in assigned[column]]
    return pd.DataFrame(out, columns=model.columns)
