"""Statistician / Utility Agent (CODE).

Proves the synthetic data is genuinely useful, climbing a hierarchy from weak
(distribution similarity) to strong (task performance / TSTR), and generating
granular per-column diagnostic reports.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agents import fairness
from agents.profiler import categorical_columns, is_numeric, numeric_columns
from models import BrokenRelationship, DiagnosticReport, UtilityResult


def _numeric_similarity(real: pd.Series, synth: pd.Series) -> float:
    r = real.dropna().astype(float)
    s = synth.dropna().astype(float)
    if r.empty or s.empty:
        return 0.0
    if len(r) > 2000:
        r = r.sample(2000, random_state=42)
    if len(s) > 2000:
        s = s.sample(2000, random_state=42)
    ks = ks_2samp(r, s).statistic  # 0 = identical distributions
    return float(max(0.0, 1.0 - ks))


def _categorical_similarity(real: pd.Series, synth: pd.Series) -> float:
    r = real.dropna().astype(str).value_counts(normalize=True)
    s = synth.dropna().astype(str).value_counts(normalize=True)
    cats = set(r.index) | set(s.index)
    tv = 0.5 * sum(abs(r.get(c, 0.0) - s.get(c, 0.0)) for c in cats)  # total variation
    return float(max(0.0, 1.0 - tv))


def distribution_fidelity(real: pd.DataFrame, synth: pd.DataFrame):
    per_col = {}
    for col in real.columns:
        if col not in synth.columns:
            continue
        if is_numeric(real[col]):
            per_col[col] = _numeric_similarity(real[col], synth[col])
        else:
            per_col[col] = _categorical_similarity(real[col], synth[col])
    fidelity = float(np.mean(list(per_col.values()))) if per_col else 0.0
    return fidelity, per_col


def find_broken_relationships(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    threshold: float = 0.15,
) -> List[BrokenRelationship]:
    """Identify column pairs with severe correlation degradation (|r_real - r_synth| >= threshold)."""
    num = [c for c in numeric_columns(real) if c in synth.columns]
    if len(num) < 2:
        return []
    rc = real[num].corr().fillna(0.0)
    sc = synth[num].corr().fillna(0.0)

    broken: List[BrokenRelationship] = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            r_val = float(rc.loc[a, b])
            s_val = float(sc.loc[a, b])
            delta = abs(r_val - s_val)
            if delta >= threshold:
                broken.append(
                    BrokenRelationship(
                        col_a=a,
                        col_b=b,
                        real_corr=round(r_val, 4),
                        synth_corr=round(s_val, 4),
                        delta=round(delta, 4),
                    )
                )
    broken.sort(key=lambda x: x.delta, reverse=True)
    return broken


def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    num = [c for c in numeric_columns(real) if c in synth.columns]
    if len(num) < 2:
        return 0.0
    rc = real[num].corr().fillna(0.0).values
    sc = synth[num].corr().fillna(0.0).values
    return float(np.abs(rc - sc).mean())


def query_equivalence(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    label_column: Optional[str] = None,
) -> float:
    """Run the same grouped mean/count query on real and synthetic data."""
    cats = [c for c in categorical_columns(real) if c in synth.columns and real[c].nunique() <= 30]
    low_card_numeric = [
        c for c in numeric_columns(real)
        if c in synth.columns and real[c].nunique() <= 30
    ]
    group_candidates = ([label_column] if label_column in cats + low_card_numeric else [])
    group_candidates += [c for c in cats + low_card_numeric if c not in group_candidates]
    nums = [c for c in numeric_columns(real) if c in synth.columns]
    if not group_candidates or not nums:
        return 0.0

    group_col = group_candidates[0]
    metric_candidates = [c for c in nums if c != group_col]
    metric_col = metric_candidates[0] if metric_candidates else nums[0]
    real_group = real.groupby(group_col, dropna=False)[metric_col].agg(["mean", "count"])
    synth_group = synth.groupby(group_col, dropna=False)[metric_col].agg(["mean", "count"])
    common = real_group.index.intersection(synth_group.index)
    if len(common) == 0:
        return 0.0

    ra, sa = real_group.loc[common], synth_group.loc[common]
    scale = max(float(real[metric_col].std()), abs(float(real[metric_col].mean())) * 0.1, 1e-9)
    mean_error = float((ra["mean"] - sa["mean"]).abs().mean() / scale)
    real_counts = ra["count"] / max(float(ra["count"].sum()), 1.0)
    synth_counts = sa["count"] / max(float(sa["count"].sum()), 1.0)
    count_error = float((real_counts - synth_counts).abs().sum() / 2.0)
    return round(float(max(0.0, min(1.0, 1.0 - 0.7 * mean_error - 0.3 * count_error))), 4)


def _make_encoder(reference: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    num = [c for c in feature_cols if is_numeric(reference[c])]
    cat = [c for c in feature_cols if not is_numeric(reference[c])]
    transformers = []
    if num:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num))
    if cat:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
        except (TypeError, ValueError):
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", ohe),
        ])
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers, remainder="drop")


def tstr(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth: pd.DataFrame,
    label_column: str,
):
    """Train-on-Synthetic / Test-on-Real vs the Train-on-Real baseline."""
    if not label_column or label_column not in real_train.columns:
        return {"utility_ratio": 0.0, "acc_real": 0.0, "acc_synth": 0.0, "task": "none", "top_features": []}
    if label_column not in synth.columns or label_column not in real_test.columns:
        return {"utility_ratio": 0.0, "acc_real": 0.0, "acc_synth": 0.0, "task": "none", "top_features": []}

    feature_cols = [c for c in real_train.columns
                    if c != label_column and c in synth.columns and c in real_test.columns]
    if not feature_cols:
        return {"utility_ratio": 0.0, "acc_real": 0.0, "acc_synth": 0.0, "task": "none", "top_features": []}

    is_classification = not is_numeric(real_train[label_column]) or \
        real_train[label_column].nunique() <= 10

    def _clean_frame(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = pd.DataFrame(index=df_in.index)
        for c in feature_cols:
            col = df_in[c]
            if is_numeric(col):
                df_out[c] = pd.to_numeric(col, errors="coerce")
            else:
                df_out[c] = col.fillna("missing").astype(str)
        return df_out

    clean_train = _clean_frame(real_train)
    clean_test = _clean_frame(real_test)
    clean_synth = _clean_frame(synth)

    union = pd.concat(
        [clean_train, clean_test, clean_synth],
        axis=0, ignore_index=True,
    )
    enc = _make_encoder(union, feature_cols)
    enc.fit(union)

    Xtr_real = enc.transform(clean_train)
    Xte_real = enc.transform(clean_test)
    Xtr_synth = enc.transform(clean_synth)

    ytr_real = real_train[label_column].astype(str)
    yte_real = real_test[label_column].astype(str)
    ytr_synth = synth[label_column].astype(str)

    # Subsample training sets to 5000 rows max to keep memory flat and prevent OOM
    if len(Xtr_real) > 5000:
        idx_r = np.random.RandomState(42).choice(len(Xtr_real), size=5000, replace=False)
        Xtr_real = Xtr_real[idx_r]
        ytr_real = ytr_real.iloc[idx_r]
    if len(Xtr_synth) > 5000:
        idx_s = np.random.RandomState(42).choice(len(Xtr_synth), size=5000, replace=False)
        Xtr_synth = Xtr_synth[idx_s]
        ytr_synth = ytr_synth.iloc[idx_s]

    top_features = feature_cols[:5]

    if is_classification:
        if ytr_synth.nunique() < 2 or ytr_real.nunique() < 2:
            return {"utility_ratio": 0.0, "acc_real": 0.0, "acc_synth": 0.0,
                    "task": "classification", "top_features": top_features}
        m_real = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42, n_jobs=2)
        m_real.fit(Xtr_real, ytr_real)
        acc_real = accuracy_score(yte_real, m_real.predict(Xte_real))

        m_synth = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42, n_jobs=2)
        m_synth.fit(Xtr_synth, ytr_synth)
        acc_synth = accuracy_score(yte_real, m_synth.predict(Xte_real))
        task = "classification"
    else:
        ytr_real = ytr_real.astype(float)
        yte_real = yte_real.astype(float)
        ytr_synth = ytr_synth.astype(float)
        m_real = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=2)
        m_real.fit(Xtr_real, ytr_real)
        acc_real = max(0.0, r2_score(yte_real, m_real.predict(Xte_real)))
        m_synth = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=2)
        m_synth.fit(Xtr_synth, ytr_synth)
        acc_synth = max(0.0, r2_score(yte_real, m_synth.predict(Xte_real)))
        task = "regression"

    ratio = float(acc_synth / acc_real) if acc_real > 0 else 0.0
    return {
        "utility_ratio": round(min(ratio, 1.0), 4),
        "acc_real": round(float(acc_real), 4),
        "acc_synth": round(float(acc_synth), 4),
        "task": task,
        "top_features": top_features,
    }


def _fairness_profile(frame: pd.DataFrame, label_column: Optional[str]):
    from models import ColumnProfile, DatasetProfile

    return DatasetProfile(
        columns=[
            ColumnProfile(name=str(column), dtype="categorical", stats={})
            for column in frame.columns
        ],
        row_count=len(frame),
        label_column=label_column,
    )


def evaluate(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth: pd.DataFrame,
    label_column: Optional[str],
    profile=None,
) -> UtilityResult:
    fidelity, per_col = distribution_fidelity(real_train, synth)
    corr_delta = correlation_delta(real_train, synth)
    q_equiv = query_equivalence(real_train, synth, label_column)
    tstr_detail = tstr(real_train, real_test, synth, label_column) if label_column else {}

    # Diagnostic auditing
    weakest = [col for col, score in per_col.items() if score < 0.80]
    broken = find_broken_relationships(real_train, synth)
    top_pred = tstr_detail.get("top_features", [])
    choke_points = [col for col in weakest if col in top_pred]

    diagnostic = DiagnosticReport(
        weakest_columns=weakest,
        broken_relationships=broken[:10],
        utility_choke_points=choke_points,
        per_column_ks={k: round(v, 4) for k, v in per_col.items()},
    )
    fairness_report = fairness.audit(
        profile or _fairness_profile(real_train, label_column),
        real_train,
        real_test,
        synth,
    )

    return UtilityResult(
        fidelity_score=round(fidelity, 4),
        per_column_distance={k: round(v, 4) for k, v in per_col.items()},
        correlation_delta=round(corr_delta, 4),
        query_equivalence=round(q_equiv, 4),
        utility_ratio=tstr_detail.get("utility_ratio", 0.0),
        tstr_detail=tstr_detail,
        diagnostic_report=diagnostic,
        fairness=fairness_report,
    )
