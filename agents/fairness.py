"""Demographic representation and subgroup utility audit.

PASS means the synthetic table keeps group shares and label rates close to the
real training table. WARN flags a shift. It does not, by itself, block release.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from agents.attack_tools import _encode_on, _encode_pair
from agents.profiler import is_numeric
from models import DatasetProfile, FairnessGroupResult, FairnessReport

_NAME_TOKENS = (
    "gender", "sex", "race", "ethnic", "age", "marital", "religion",
    "nationality", "native_country",
)
_LABEL_GAP_WARN = 0.05
_TVD_WARN = 0.10
_UTILITY_DROP_WARN = 0.15
_MIN_GROUP = 20


def demographic_columns(profile: DatasetProfile, frame: pd.DataFrame) -> List[str]:
    found = []
    for column in profile.columns:
        name = column.name
        if name not in frame.columns or name == profile.label_column:
            continue
        lowered = name.lower()
        if any(token in lowered for token in _NAME_TOKENS):
            found.append(name)
    return found


def _group_labels(series: pd.Series, column: str) -> pd.Series:
    if is_numeric(series) and "age" in column.lower():
        numeric = pd.to_numeric(series, errors="coerce")
        bands = (numeric // 10 * 10).astype("Int64").astype(str)
        return bands.fillna("missing")
    return series.fillna("missing").astype(str).str.strip()


def _positive_label(labels: pd.Series) -> str:
    counts = labels.fillna("missing").astype(str).value_counts()
    if counts.empty:
        return "missing"
    if len(counts) == 2:
        return str(counts.index[-1])
    return str(counts.index[0])


def _tvd(real_share: pd.Series, synth_share: pd.Series) -> float:
    groups = set(real_share.index) | set(synth_share.index)
    return float(0.5 * sum(abs(float(real_share.get(group, 0.0)) - float(synth_share.get(group, 0.0))) for group in groups))


def _subgroup_accuracy(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth: pd.DataFrame,
    label: str,
    groups: pd.Series,
) -> Dict[str, float]:
    features = [column for column in real_train.columns if column != label and column in synth.columns and column in real_test.columns]
    if not features or real_train[label].nunique(dropna=True) < 2:
        return {}
    train = real_train if len(real_train) <= 1500 else real_train.sample(1500, random_state=42)
    synth_fit = synth if len(synth) <= 1500 else synth.sample(1500, random_state=42)
    holdout = real_test if len(real_test) <= 1500 else real_test.sample(1500, random_state=42)
    try:
        _real_enc, synth_enc, used = _encode_pair(train, synth_fit, features)
        hold_enc = _encode_on(train, synth_fit, holdout, used)
    except (ValueError, KeyError):
        return {}
    if synth_enc.shape[1] == 0 or len(hold_enc) != len(holdout):
        return {}
    y_synth = synth_fit[label].fillna("missing").astype(str)
    y_hold = holdout[label].fillna("missing").astype(str)
    if y_synth.nunique() < 2:
        return {}
    model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42, n_jobs=1)
    model.fit(synth_enc, y_synth)
    known = set(model.classes_)
    mask = y_hold.isin(known).to_numpy()
    if int(mask.sum()) < _MIN_GROUP:
        return {}
    scored = pd.DataFrame({
        "pred": model.predict(hold_enc[mask]),
        "actual": y_hold.to_numpy()[mask],
        "group": groups.reindex(holdout.index).fillna("missing").to_numpy()[mask],
    })
    scores = {"__overall__": float((scored["pred"] == scored["actual"]).mean())}
    for group, part in scored.groupby("group"):
        if len(part) < _MIN_GROUP:
            continue
        scores[str(group)] = float((part["pred"] == part["actual"]).mean())
    return scores


def audit(
    profile: DatasetProfile,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth: pd.DataFrame,
) -> FairnessReport:
    attributes = [name for name in demographic_columns(profile, real_train) if name in synth.columns]
    if not attributes:
        return FairnessReport(
            status="PASS",
            summary="No demographic attributes detected.",
        )

    label = profile.label_column if profile.label_column in real_train.columns else None
    positive = _positive_label(real_train[label]) if label else None
    subgroup_scores: Dict[str, float] = {}
    if label and label in synth.columns and label in real_test.columns:
        # Slice on the first demographic so one model serves every attribute that shares those rows.
        first_groups = _group_labels(real_test[attributes[0]], attributes[0])
        subgroup_scores = _subgroup_accuracy(real_train, real_test, synth, label, first_groups)

    groups: List[FairnessGroupResult] = []
    representation: Dict[str, float] = {}
    warned = False
    for attribute in attributes:
        real_groups = _group_labels(real_train[attribute], attribute)
        synth_groups = _group_labels(synth[attribute], attribute)
        real_share = real_groups.value_counts(normalize=True)
        synth_share = synth_groups.value_counts(normalize=True)
        tvd = _tvd(real_share, synth_share)
        representation[attribute] = round(tvd, 4)
        if tvd > _TVD_WARN:
            warned = True
        names = list(dict.fromkeys(list(real_share.index) + list(synth_share.index)))
        for group in names[:8]:
            real_rate = float(real_share.get(group, 0.0))
            synth_rate = float(synth_share.get(group, 0.0))
            label_gap = 0.0
            if label and positive is not None and label in synth.columns:
                real_mask = real_groups == group
                synth_mask = synth_groups == group
                if int(real_mask.sum()) >= _MIN_GROUP and int(synth_mask.sum()) >= _MIN_GROUP:
                    real_positive = float((real_train.loc[real_mask, label].astype(str) == positive).mean())
                    synth_positive = float((synth.loc[synth_mask, label].astype(str) == positive).mean())
                    label_gap = abs(real_positive - synth_positive)
            utility = subgroup_scores.get(str(group)) if attribute == attributes[0] else None
            overall = subgroup_scores.get("__overall__")
            status = "PASS"
            if abs(real_rate - synth_rate) > _TVD_WARN or label_gap > _LABEL_GAP_WARN:
                status = "WARN"
            if utility is not None and overall is not None and real_rate < 0.4 and utility < overall - _UTILITY_DROP_WARN:
                status = "WARN"
            if status == "WARN":
                warned = True
            groups.append(FairnessGroupResult(
                attribute=attribute,
                group=str(group),
                real_share=round(real_rate, 4),
                synth_share=round(synth_rate, 4),
                label_gap=round(label_gap, 4),
                subgroup_utility=None if utility is None else round(utility, 4),
                status=status,
            ))

    status = "WARN" if warned else "PASS"
    summary = (
        "Demographic representation or a subgroup label rate shifted past the review line."
        if warned
        else "Group shares and label rates stay within the review line."
    )
    return FairnessReport(
        status=status,
        attributes=attributes,
        representation_tvd=representation,
        groups=groups,
        summary=summary,
    )
