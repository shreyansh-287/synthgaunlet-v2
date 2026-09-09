"""Deterministic attack tools (CODE - pure math, no LLM).

The Attacker *agent* decides WHERE and HOW to attack; these functions do the
math so the numbers are trustworthy. Sanity property: feeding an identical copy
of the real data as "synthetic" must yield ~100% re-identification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agents.profiler import is_numeric


def _build_encoder(reference: pd.DataFrame, columns: List[str]) -> ColumnTransformer:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    num_cols = [c for c in columns if is_numeric(reference[c])]
    cat_cols = [c for c in columns if not is_numeric(reference[c])]
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
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
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def _clean_features(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_out = pd.DataFrame(index=df_in.index)
    for c in cols:
        col = df_in[c]
        if is_numeric(col):
            df_out[c] = pd.to_numeric(col, errors="coerce")
        else:
            df_out[c] = col.fillna("missing").astype(str)
    return df_out


def _encode_pair(
    real: pd.DataFrame, synth: pd.DataFrame, columns: List[str]
):
    """Fit a consistent transformer on the union of real+synth, transform both."""
    columns = [c for c in columns if c in real.columns and c in synth.columns]
    clean_real = _clean_features(real, columns)
    clean_synth = _clean_features(synth, columns)
    union = pd.concat([clean_real, clean_synth], axis=0, ignore_index=True)
    enc = _build_encoder(union, columns)
    enc.fit(union)
    return enc.transform(clean_real), enc.transform(clean_synth), columns


def run_reidentification(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    quasi_identifiers: List[str],
    threshold: float,
    subset: Optional[List[int]] = None,
    k_anonymity: int = 5,
    real_holdout: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Re-identification risk.

    Two modes:

    * DCR-baseline (used in the pipeline, when `real_holdout` is provided):
      A synthetic row is a genuine re-identification only if it lands
      *abnormally close* to a real training record -- closer than a genuine
      held-out real person typically is. We take the 5th-percentile
      distance-to-closest-record (DCR) among held-out real people as the natural
      baseline of "how identifiable a real person is", then report the EXCESS
      fraction of synthetic rows that beat that bar and match a rare (non
      k-anonymous) individual. Safe synthetic data carries ~0 excess risk; an
      exact copy carries ~1.

    * Threshold mode (fallback, no holdout): flag near-copies within a scaled
      distance threshold, k-anonymity aware.
    """
    cols = [c for c in quasi_identifiers if c in real.columns and c in synth.columns]
    if not cols:
        return {"re_id_rate": 0.0, "flagged_examples": [], "n_checked": 0, "columns": []}

    real_enc, synth_enc, used = _encode_pair(real, synth, cols)
    if real_enc.shape[1] == 0:
        return {"re_id_rate": 0.0, "flagged_examples": [], "n_checked": 0, "columns": used}

    # k-anonymity: how many real people share each quasi-identifier combination.
    # Numeric quasi-identifiers are GENERALIZED into bins first (standard
    # k-anonymity), so "rare" reflects genuine uniqueness -- not merely a unique
    # exact age -- and coincidental matches to people in a crowd don't count.
    real_keys = _kanon_keys(real, used)
    combo_counts = real_keys.value_counts(dropna=False)
    real_is_rare = real_keys.map(lambda key: combo_counts.get(key, 0) < k_anonymity).to_numpy()

    nn = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(real_enc)
    distances, indices = nn.kneighbors(synth_enc)
    distances = distances.ravel()
    indices = indices.ravel()
    matched_rare = real_is_rare[indices]

    if real_holdout is not None and len(real_holdout) >= 20:
        # DCR baseline from genuine held-out real people.
        hold_enc = _encode_on(real, synth, real_holdout, used)
        hold_dist, _ = nn.kneighbors(hold_enc)
        cut = float(np.quantile(hold_dist.ravel(), 0.05))
        baseline = 0.05  # by construction 5% of real holdout beat the cut
        near = (distances <= cut) & matched_rare
        re_id_rate = float(max(0.0, near.mean() - baseline))
        flagged_mask = near
    else:
        scaled_threshold = threshold * np.sqrt(real_enc.shape[1])
        flagged_mask = (distances < scaled_threshold) & matched_rare
        re_id_rate = float(flagged_mask.mean())

    flagged_examples: List[Dict[str, Any]] = []
    for si in np.where(flagged_mask)[0][:5]:
        ri = int(indices[si])
        flagged_examples.append(
            {
                "distance": round(float(distances[si]), 4),
                "quasi_identifiers": {c: str(synth[c].iloc[si]) for c in used},
                "real_index": ri,
                "is_rare_in_real": bool(matched_rare[si]),
            }
        )

    return {
        "re_id_rate": re_id_rate,
        "flagged_examples": flagged_examples,
        "n_checked": len(synth),
        "columns": used,
    }


def _encode_on(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target: pd.DataFrame,
    columns: List[str],
):
    """Fit the union encoder (real+synth) and transform `target` consistently."""
    clean_real = _clean_features(real, columns)
    clean_synth = _clean_features(synth, columns)
    clean_target = _clean_features(target, columns)
    union = pd.concat([clean_real, clean_synth], axis=0, ignore_index=True)
    enc = _build_encoder(union, columns)
    enc.fit(union)
    return enc.transform(clean_target)


def _kanon_keys(df: pd.DataFrame, columns: List[str], n_bins: int = 10) -> pd.Series:
    """Build k-anonymity group keys, generalizing numeric columns into bins and handling nulls safely."""
    parts = []
    for c in columns:
        col = df[c]
        if is_numeric(col):
            try:
                binned = pd.qcut(col.dropna().astype(float), q=n_bins, duplicates="drop")
                parts.append(binned.reindex(df.index).astype(str).fillna("<NA>"))
            except (ValueError, TypeError):
                parts.append(col.astype(str).fillna("<NA>"))
        else:
            parts.append(col.astype(str).fillna("<NA>"))
    keys = parts[0]
    for p in parts[1:]:
        keys = keys.str.cat(p, sep="|", na_rep="<NA>")
    return keys


def run_membership_inference(
    real_in: pd.DataFrame,
    real_out: pd.DataFrame,
    synth: pd.DataFrame,
    features: List[str],
) -> Dict[str, Any]:
    """Distance-to-synthetic membership attack.

    IN  = records used to build the profile/generate (should look closer to synth)
    OUT = held-out records the generator never saw
    If a classifier can tell IN from OUT using distance-to-nearest-synth, the
    synthetic data is leaking membership. Accuracy ~0.5 == safe.
    """
    cols = [c for c in features if c in synth.columns
            and c in real_in.columns and c in real_out.columns]
    if not cols or len(real_in) < 10 or len(real_out) < 10:
        return {"membership_accuracy": 0.5, "auc": 0.5, "n_test": 0}

    # Cap n to 2500 max per split to prevent OOM on 10,000+ row runs
    n = min(len(real_in), len(real_out), 2500)
    real_in = real_in.sample(n=n, random_state=42)
    real_out = real_out.sample(n=n, random_state=42)

    # Subsample synth if > 5000 rows to keep distance matrix computation bounded
    synth_target = synth if len(synth) <= 5000 else synth.sample(n=5000, random_state=42)

    combined_real = pd.concat([real_in, real_out], axis=0, ignore_index=True)
    real_enc, synth_enc, _ = _encode_pair(combined_real, synth_target, cols)

    nn = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(synth_enc)
    dist, _ = nn.kneighbors(real_enc)
    dist = dist.ravel().reshape(-1, 1)

    labels = np.array([1] * n + [0] * n)  # 1 = IN

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(labels))
    split = int(0.7 * len(labels))
    train_idx, test_idx = perm[:split], perm[split:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(dist[train_idx], labels[train_idx])
    acc = float(clf.score(dist[test_idx], labels[test_idx]))

    return {
        "membership_accuracy": acc,
        "mean_dist_in": float(dist[labels == 1].mean()),
        "mean_dist_out": float(dist[labels == 0].mean()),
        "n_test": int(len(test_idx)),
    }


def run_outlier_linkage(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    columns: List[str],
    rarity_quantile: float = 0.05,
) -> Dict[str, Any]:
    """Target the rarest real quasi-identifier combinations -- the highest
    re-identification risk -- and measure how often synth reproduces them."""
    cols = [c for c in columns if c in real.columns and c in synth.columns]
    if not cols:
        return {"outlier_linkage_rate": 0.0, "n_outliers": 0}

    def _row_keys(df: pd.DataFrame, c_list: List[str]) -> pd.Series:
        parts = [df[c].fillna("<NA>").astype(str) for c in c_list]
        res = parts[0]
        for p in parts[1:]:
            res = res.str.cat(p, sep="|", na_rep="<NA>")
        return res

    real_keys = _row_keys(real, cols)
    synth_keys = set(_row_keys(synth, cols))

    counts = real_keys.value_counts()
    # "Outliers" = combos that occur very rarely in the real data.
    rare_combos = counts[counts <= max(1, counts.quantile(rarity_quantile))].index
    if len(rare_combos) == 0:
        return {"outlier_linkage_rate": 0.0, "n_outliers": 0}

    reproduced = sum(1 for combo in rare_combos if combo in synth_keys)
    return {
        "outlier_linkage_rate": float(reproduced / len(rare_combos)),
        "n_outliers": int(len(rare_combos)),
        "n_reproduced": int(reproduced),
    }
