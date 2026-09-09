"""Report Writer & Publisher (LLM narrative + CODE export).

Composes the human-readable Dual Scorecard. CSV export today; BigQuery publish
is stubbed for the GCP-native path (roadmap).
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

import llm
from config import OUTPUT_DIR, get_settings
from models import DualScorecard, RoundResult, Thresholds

logger = logging.getLogger("forge.reporter")


def build_dual_scorecard(
    rounds: List[RoundResult],
    status: str,
    thresholds: Thresholds,
) -> DualScorecard:
    final = rounds[-1]
    return DualScorecard(
        status=status,
        utility=final.utility,
        privacy=final.attack,
        thresholds=thresholds,
        passed=status == "SUCCESS" and final.passed,
        rounds=len(rounds),
    )


def build_scorecard(
    rounds: List[RoundResult],
    status: str,
    thresholds: Thresholds,
    domain: str,
    difficulty: str = "",
    difficulty_detail: str = "",
) -> str:
    if not rounds:
        return "No rounds were executed."
    final = rounds[-1]
    u, a = final.utility, final.attack

    deterministic = _deterministic_scorecard(final, status, thresholds)
    if difficulty:
        deterministic += (
            f"\n\nDATASET DIFFICULTY: {difficulty.upper()}\n  {difficulty_detail}"
        )

    settings = get_settings()
    narrative = llm.generate_text(
        "Write a crisp 3-4 sentence executive summary of this synthetic-data run "
        "for a compliance officer. Be concrete and confident, cite the numbers.\n\n"
        f"Domain: {domain}\nStatus: {status}\nRounds: {len(rounds)}\n"
        f"Fidelity: {u.fidelity_score}, Utility ratio (TSTR): {u.utility_ratio}, "
        f"Query equivalence: {u.query_equivalence}\n"
        f"Re-identification: {a.re_id_rate}, Membership accuracy: {a.membership_accuracy}",
        model=settings.reporter_model,
        temperature=0.6,
    )

    if narrative:
        return f"{deterministic}\n\nEXECUTIVE SUMMARY\n{narrative}"
    return deterministic


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _deterministic_scorecard(final: RoundResult, status: str, t: Thresholds) -> str:
    u, a = final.utility, final.attack
    ok = "PASS"
    bad = "FAIL"
    lines = [
        "================= DUAL SCORECARD =================",
        "USEFUL (proven, not claimed)",
        f"  Distribution fidelity:   {_pct(u.fidelity_score)}  "
        f"[{ok if u.fidelity_score >= t.fidelity_min else bad}]",
        f"  Utility ratio (TSTR):    {_pct(u.utility_ratio)}  "
        f"[{ok if u.utility_ratio >= t.tstr_min else bad}] "
        f"(acc_synth={u.tstr_detail.get('acc_synth')}, acc_real={u.tstr_detail.get('acc_real')})",
        f"  Query equivalence:       {_pct(u.query_equivalence)}  "
        f"[{ok if u.query_equivalence >= t.query_equivalence_min else bad}]",
        f"  Correlation delta:       {u.correlation_delta}",
        "SAFE (proven by attack)",
        f"  Re-identification risk:  {_pct(a.re_id_rate)}  "
        f"[{ok if a.re_id_rate <= t.reid_max else bad}]",
        f"  Membership leakage:      {_pct(a.membership_accuracy)}  "
        f"[{ok if a.membership_accuracy <= t.membership_max else bad}] (coin flip = safe)",
        f"  Outlier linkage:         {_pct(a.outlier_linkage_rate)}",
        f"VERDICT: {'USEFUL *AND* SAFE' if status == 'SUCCESS' else 'DID NOT CONVERGE'}",
        "=================================================",
    ]
    return "\n".join(lines)


def export_csv(synth: pd.DataFrame, run_id: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{run_id}.csv"
    synth.to_csv(path, index=False)
    logger.info("Exported synthetic data to %s", path)
    return str(path)


def publish_bigquery(synth: pd.DataFrame, table_id: str) -> str:
    """Publish a validated synthetic frame to a BigQuery table."""
    from google.cloud import bigquery

    job = bigquery.Client().load_table_from_dataframe(synth, table_id)
    job.result()
    return table_id
