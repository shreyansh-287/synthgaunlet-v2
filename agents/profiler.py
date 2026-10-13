"""Statistical Profiler (CODE + LLM Semantic Analyst).

Extracts a statistical fingerprint of the source data. Raw rows are NEVER
handed to the LLM -- only these aggregates.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

import llm
from config import get_settings
from models import ColumnProfile, DatasetProfile, SemanticProfile

logger = logging.getLogger("forge.profiler")

# Columns with more distinct values than this ratio are treated as numeric-ish
# identifiers rather than categoricals for correlation purposes.
_MAX_CATEGORICAL_CARDINALITY = 50

_ANALYST_SYSTEM = (
    "You are an expert Data Profiler and Semantic Schema Analyst. Analyze the "
    "provided statistical aggregates of a tabular dataset (NEVER raw rows) to infer "
    "functional dependencies, high-risk quasi-identifier combinations, business rules, "
    "and the optimal conditional generation ordering. Respond in valid JSON adhering to the requested schema."
)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def infer_dtype(series: pd.Series) -> str:
    return "numeric" if is_numeric(series) else "categorical"


def _numeric_stats(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if s.empty:
        return {"count": 0}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
    }


def _categorical_stats(series: pd.Series, top: int = 20) -> dict:
    s = series.dropna().astype(str)
    freqs = (s.value_counts(normalize=True)).head(top)
    return {
        "count": int(s.count()),
        "cardinality": int(series.nunique()),
        "top_values": {str(k): round(float(v), 5) for k, v in freqs.items()},
    }


def profile_dataframe(
    df: pd.DataFrame,
    *,
    quasi_identifiers: Optional[List[str]] = None,
    sensitive_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
) -> DatasetProfile:
    qi_names = list(quasi_identifiers or [])
    quasi_identifiers = set(qi_names)
    sensitive_columns = set(sensitive_columns or [])

    columns: List[ColumnProfile] = []
    for name in df.columns:
        dtype = infer_dtype(df[name])
        stats = (
            _numeric_stats(df[name])
            if dtype == "numeric"
            else _categorical_stats(df[name])
        )
        columns.append(
            ColumnProfile(
                name=name,
                dtype=dtype,
                stats=stats,
                missing_rate=round(float(df[name].isna().mean()), 5),
                is_quasi_identifier=name in quasi_identifiers,
                is_sensitive=name in sensitive_columns,
            )
        )

    correlations = _pairwise_correlations(df)
    missingness = {name: round(float(df[name].isna().mean()), 5) for name in df.columns}

    return DatasetProfile(
        columns=columns,
        correlations=correlations,
        missingness=missingness,
        rare_combinations=_rare_combinations(df, qi_names),
        row_count=int(len(df)),
        label_column=label_column,
    )


def _rare_combinations(df: pd.DataFrame, quasi_identifiers: List[str]) -> List[dict]:
    columns = [column for column in quasi_identifiers if column in df.columns]
    if len(columns) < 2 or df.empty:
        return []
    counts = df[columns].astype(str).fillna("<missing>").value_counts()
    cutoff = max(1, int(np.ceil(len(df) * 0.01)))
    rare = counts[counts <= cutoff].head(20)
    return [
        {"columns": columns, "frequency": int(count), "frequency_rate": round(float(count / len(df)), 5)}
        for count in rare.tolist()
    ]


def _pairwise_correlations(df: pd.DataFrame) -> dict:
    numeric_cols = [c for c in df.columns if is_numeric(df[c])]
    if len(numeric_cols) < 2:
        return {}
    corr = df[numeric_cols].corr(method="pearson").fillna(0.0)
    out = {}
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            out[f"{a}|{b}"] = round(float(corr.loc[a, b]), 4)
    return out


def analyze_profile(
    profile: DatasetProfile,
    business_rules: Optional[str] = None,
) -> SemanticProfile:
    """LLM Profile Analyst: extracts functional dependencies, high-risk combinations,

    constraints, and generation ordering strictly from aggregated metadata (NO raw rows).
    """
    settings = get_settings()
    col_summary = []
    for c in profile.columns:
        tags = []
        if c.is_quasi_identifier:
            tags.append("QI")
        if c.is_sensitive:
            tags.append("SENSITIVE")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        if c.dtype == "numeric":
            st = c.stats
            col_summary.append(
                f"- {c.name}{tag_str} (numeric): min={st.get('min')}, max={st.get('max')}, "
                f"mean={st.get('mean')}, std={st.get('std')}"
            )
        else:
            st = c.stats
            top_vals = list(st.get("top_values", {}).keys())[:4]
            col_summary.append(
                f"- {c.name}{tag_str} (categorical): cardinality={st.get('cardinality')}, sample_top={top_vals}"
            )

    high_corrs = [
        f"{pair}: r={val}" for pair, val in profile.correlations.items() if abs(val) >= 0.35
    ]

    prompt = (
        "Analyze this dataset's aggregate profile:\n"
        f"Row count: {profile.row_count}\n"
        f"Label column: {profile.label_column or 'None'}\n"
        f"Columns:\n" + "\n".join(col_summary) + "\n\n"
        f"High Correlations (|r| >= 0.35):\n" + (", ".join(high_corrs) if high_corrs else "None") + "\n\n"
        f"Rare QI Combinations Count: {len(profile.rare_combinations)}\n"
        f"Business Rules: {business_rules or 'Standard domain consistency'}\n\n"
        "Provide:\n"
        "1. functional_dependencies: map of child->parent column for dependent hierarchies (e.g. JobRole -> Department)\n"
        "2. high_risk_qi_combos: list of column groups that form unique identifying combinations\n"
        "3. business_constraints: list of key constraints (e.g. non-negative numeric bounds)\n"
        "4. conditional_generation_order: list of all columns in recommended topological order (parents before children, label last)\n"
        "5. analyst_notes: brief 2-sentence summary of findings."
    )

    result = llm.generate_structured(
        prompt,
        SemanticProfile,
        model=settings.generator_model,
        system_instruction=_ANALYST_SYSTEM,
        temperature=0.2,
        role="profiler",
        purpose="Infer dependencies, risky column groups, and the generation order from the aggregate profile.",
    )

    if result is not None and isinstance(result, SemanticProfile):
        return result

    # Deterministic fallback heuristic
    all_col_names = [c.name for c in profile.columns]
    qi_cols = [c.name for c in profile.columns if c.is_quasi_identifier]
    order = [c for c in all_col_names if c != profile.label_column]
    if profile.label_column and profile.label_column in all_col_names:
        order.append(profile.label_column)

    return SemanticProfile(
        functional_dependencies={},
        high_risk_qi_combos=[qi_cols] if len(qi_cols) >= 2 else [],
        business_constraints=["Maintain positive values for counts and monetary attributes"],
        conditional_generation_order=order,
        analyst_notes="Deterministic semantic profile fallback applied.",
    )


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_numeric(df[c])]


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not is_numeric(df[c])]
