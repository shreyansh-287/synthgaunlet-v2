"""Adversarial multi-agent pipeline: the Forge run.

Ties the cast together per STORY.md with enterprise Vertex AI optimizations:
  Profiler (code + semantic analyst) ->
  Generator (Forge multi-candidate beam search) ->
  Statistician (Auditor deep diagnostics) ->
  Attacker (Challenger adaptive tool loop) ->
  Orchestrator (Arbiter Pareto selector & deterministic release gate)
  --loop-- until useful AND safe.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import certificate
import failure_report
import llm
import reporter
from agents import attacker, dp_engine, generator, orchestrator, profiler, statistician
from agents.profiler import profile_dataframe
from config import get_settings
from data_source import get_data_source
from models import (
    AIUsageStep,
    CandidateEvaluation,
    ForgeRequest,
    ForgeResponse,
    RescueAttempt,
    RoundResult,
    RunBlackboard,
    RunStartResponse,
    RunStatusResponse,
    Thresholds,
)

logger = logging.getLogger("forge.pipeline")

# In-memory registry of the latest synthetic frames, keyed by run_id (for export).
_RUN_CACHE: Dict[str, pd.DataFrame] = {}
# Passing candidates only, keyed by run_id then candidate_id. The slider may switch among these.
_PASSING_FRAMES: Dict[str, Dict[str, pd.DataFrame]] = {}
_RUN_STATES: Dict[str, dict] = {}
_RUN_LOCK = threading.RLock()
_RUN_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="synthgauntlet")

ProgressFn = Callable[[dict], None]


def _record_event(run_id: str, event: dict) -> None:
    with _RUN_LOCK:
        state = _RUN_STATES.setdefault(run_id, {"status": "QUEUED", "progress": []})
        if "progress" not in state:
            state["progress"] = []
        state["progress"].append(event)
        state["progress"] = state["progress"][-250:]
        state["latest_event"] = event
        if event.get("type") == "done":
            state["status"] = event["response"].get("status", "FAILED")
            state["response"] = event["response"]
        elif event.get("type") == "error":
            state["status"] = "FAILED"
            state["error"] = event.get("message", "run failed")
        elif state.get("status") == "QUEUED":
            state["status"] = "RUNNING"


def abort_run(run_id: str) -> bool:
    """Abort an active run. Instantly terminates execution and purges cached frames so zero synthetic data is exported."""
    with _RUN_LOCK:
        state = _RUN_STATES.setdefault(run_id, {"status": "ABORTED", "progress": []})
        state["status"] = "ABORTED"
        state["error"] = "Execution aborted by user."
        if "progress" not in state:
            state["progress"] = []
        _RUN_CACHE.pop(run_id, None)
        _PASSING_FRAMES.pop(run_id, None)
        _record_event(run_id, {"type": "aborted", "message": "Execution aborted by user. Zero data generated."})
        return True


def _is_aborted(run_id: str) -> bool:
    with _RUN_LOCK:
        st = _RUN_STATES.get(run_id, {})
        return st.get("status") == "ABORTED"


def get_run_status(run_id: str) -> Optional[RunStatusResponse]:
    with _RUN_LOCK:
        state = _RUN_STATES.get(run_id)
        if state is None:
            return None
        response = state.get("response")
        return RunStatusResponse(
            run_id=run_id,
            status=state["status"],
            progress=state.get("progress", []),
            response=response,
            error=state.get("error"),
        )


def _run_in_background(req: ForgeRequest, run_id: str) -> None:
    try:
        run_forge(req, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background run %s failed", run_id)
        _record_event(run_id, {"type": "error", "message": str(exc)})


def start_run(req: ForgeRequest) -> RunStartResponse:
    run_id = uuid.uuid4().hex[:12]
    with _RUN_LOCK:
        _RUN_STATES[run_id] = {"status": "QUEUED", "progress": []}
    _RUN_EXECUTOR.submit(_run_in_background, req, run_id)
    return RunStartResponse(run_id=run_id, status="QUEUED")


def get_run_frame(run_id: str) -> Optional[pd.DataFrame]:
    return _RUN_CACHE.get(run_id)


def get_passing_frame(run_id: str, candidate_id: str) -> Optional[pd.DataFrame]:
    """Return a candidate table only when that candidate passed the release gate."""
    frames = _PASSING_FRAMES.get(run_id) or {}
    frame = frames.get(candidate_id)
    if frame is None or frame.empty:
        return None
    return frame


def activate_passing_candidate(run_id: str, candidate_id: str) -> Optional[pd.DataFrame]:
    """Make a certified candidate the table behind export and BigQuery."""
    frame = get_passing_frame(run_id, candidate_id)
    if frame is None:
        return None
    with _RUN_LOCK:
        state = _RUN_STATES.get(run_id)
        stored = state.get("response") if state else None
        if isinstance(stored, dict):
            stored["active_candidate_id"] = candidate_id
        elif stored is not None and hasattr(stored, "active_candidate_id"):
            stored.active_candidate_id = candidate_id
    _RUN_CACHE[run_id] = frame
    reporter.export_csv(frame, run_id)
    return frame


def compute_column_distributions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    max_cols: int = 12,
) -> Optional[Dict[str, Any]]:
    """Compute normalized histogram and categorical distributions for Real vs Synthetic comparison."""
    if real_df is None or synth_df is None or real_df.empty or synth_df.empty:
        return None
    try:
        res: Dict[str, Any] = {"columns": [], "details": {}}
        common_cols = [c for c in real_df.columns if c in synth_df.columns]
        for col in common_cols[:max_cols]:
            try:
                r_s = real_df[col].dropna()
                s_s = synth_df[col].dropna()
                if len(r_s) == 0 or len(s_s) == 0:
                    continue
                is_num = pd.api.types.is_numeric_dtype(r_s) and r_s.nunique() > 10
                if is_num:
                    c_min = float(min(r_s.min(), s_s.min()))
                    c_max = float(max(r_s.max(), s_s.max()))
                    if c_min == c_max:
                        c_max += 1.0
                    bins = np.linspace(c_min, c_max, 11)
                    r_counts, _ = np.histogram(r_s, bins=bins)
                    s_counts, _ = np.histogram(s_s, bins=bins)
                    r_total = max(1, len(r_s))
                    s_total = max(1, len(s_s))
                    labels = [
                        f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                        if (c_max - c_min < 100)
                        else f"{int(bins[i])}-{int(bins[i+1])}"
                        for i in range(10)
                    ]
                    r_pct = [round(float(c) / r_total * 100, 1) for c in r_counts]
                    s_pct = [round(float(c) / s_total * 100, 1) for c in s_counts]
                    res["columns"].append(col)
                    res["details"][col] = {
                        "type": "numeric",
                        "labels": labels,
                        "real_pct": r_pct,
                        "synth_pct": s_pct,
                        "real_stats": {
                            "mean": round(float(r_s.mean()), 2),
                            "std": round(float(r_s.std()), 2),
                            "min": round(c_min, 2),
                            "max": round(c_max, 2),
                        },
                        "synth_stats": {
                            "mean": round(float(s_s.mean()), 2),
                            "std": round(float(s_s.std()), 2),
                            "min": round(float(s_s.min()), 2),
                            "max": round(float(s_s.max()), 2),
                        },
                    }
                else:
                    top_cats = r_s.astype(str).value_counts().head(7).index.tolist()
                    r_str = r_s.astype(str).apply(lambda x: x if x in top_cats else "Other")
                    s_str = s_s.astype(str).apply(lambda x: x if x in top_cats else "Other")
                    categories = list(top_cats)
                    if (r_str == "Other").any() or (s_str == "Other").any():
                        if "Other" not in categories:
                            categories.append("Other")
                    r_vc = r_str.value_counts(normalize=True) * 100
                    s_vc = s_str.value_counts(normalize=True) * 100
                    r_pct = [round(float(r_vc.get(c, 0.0)), 1) for c in categories]
                    s_pct = [round(float(s_vc.get(c, 0.0)), 1) for c in categories]
                    res["columns"].append(col)
                    res["details"][col] = {
                        "type": "categorical",
                        "labels": [str(c)[:18] for c in categories],
                        "real_pct": r_pct,
                        "synth_pct": s_pct,
                        "real_stats": {
                            "top": str(r_s.mode().iloc[0]) if not r_s.empty else "N/A",
                            "unique": int(r_s.nunique()),
                        },
                        "synth_stats": {
                            "top": str(s_s.mode().iloc[0]) if not s_s.empty else "N/A",
                            "unique": int(s_s.nunique()),
                        },
                    }
            except Exception:
                continue
        return res if res["columns"] else None
    except Exception as exc:
        logger.warning("compute_column_distributions failed gracefully: %s", exc)
        return None


def _resolve_columns(req: ForgeRequest, meta) -> Dict[str, list]:
    qis = req.quasi_identifiers if req.quasi_identifiers is not None else meta.quasi_identifiers
    sens = req.sensitive_columns if req.sensitive_columns is not None else meta.sensitive_columns
    label = req.label_column or meta.label_column
    return {"quasi_identifiers": qis or [], "sensitive_columns": sens or [], "label": label}


def _assess_difficulty(n_rows: int, quasi_identifiers: list) -> Dict[str, str]:
    """Explain, in plain English, how hard this dataset is to anonymise."""
    n_qi = len(quasi_identifiers)
    if n_qi <= 2 or n_rows < 1000:
        level = "hard"
        detail = (
            f"Only {n_qi} quasi-identifier(s) across {n_rows:,} rows. With such a "
            "small attack surface, protecting rare individuals without blurring the "
            "data is a genuine tightrope -- the engine may honestly refuse to certify."
        )
    elif n_qi <= 3 or n_rows < 5000:
        level = "moderate"
        detail = (
            f"{n_qi} quasi-identifiers across {n_rows:,} rows. A workable balance of "
            "utility and privacy usually exists; may take a couple of rounds."
        )
    else:
        level = "easy"
        detail = (
            f"{n_qi} quasi-identifiers across {n_rows:,} rows -- a rich, redundant "
            "attack surface that is comfortably anonymised while staying useful."
        )
    return {"difficulty": level, "difficulty_detail": detail}


def _certification(passed: bool, rescued: bool) -> str:
    if passed and rescued:
        return "CERTIFIED_AFTER_RESCUE"
    if passed:
        return "CERTIFIED"
    return "NOT_CERTIFIED"


def _auto_heal(
    *,
    emit: ProgressFn,
    run_id: str,
    profile,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth: pd.DataFrame,
    plan,
    cols: Dict[str, list],
    n_rows: int,
    thresholds: Thresholds,
    generation_order: Optional[list],
    prior: Optional[dict],
    max_rounds: int,
    best_score: float,
    failure_brief: str = "",
) -> Optional[dict]:
    """Three-step rescue ladder. Every attempt is scored on the original real split.

    Returns None when the user aborts mid-rescue.
    """
    steps = (
        (1, "tighten", "Search privacy-safe redraws, remove low-DCR rows, and suppress singleton QI combinations."),
        (2, "generalize", "Bucket numeric quasi-identifiers, collapse categories under 2% to Other, and winsorize tails."),
        (3, "second_generation", "Resample from the best failed candidate with stronger perturbation."),
    )
    advice = ""
    if failure_brief:
        advice = llm.advise(
            "orchestrator",
            "Explain which failed gates Auto-Heal should focus on.",
            (
                "The ordinary rounds missed the release gate. The rescue ladder still runs in code, "
                "and only a table that passes every gate can be released. "
                f"Latest scores: {failure_brief}"
            ),
            model=get_settings().orchestrator_model,
        )
    emit({
        "type": "rescue_started",
        "attempts": len(steps),
        "message": "Release gate failed. Auto-Heal rescue ladder started."
        + (f" {advice}" if advice else ""),
    })
    numeric_qis = [
        c for c in cols["quasi_identifiers"]
        if c in real_train.columns and c in profiler.numeric_columns(real_train)
    ]
    best_synth = synth.copy()
    origin_plan = plan
    attempts: List[RescueAttempt] = []
    extra_rounds: List[RoundResult] = []
    passed = False
    current = synth

    for attempt, name, message in steps:
        if _is_aborted(run_id):
            emit({"type": "rescue_finished", "final_status": "NOT_CERTIFIED", "attempts_run": len(attempts), "passed": False})
            return None
        emit({
            "type": "rescue_attempt",
            "attempt": attempt,
            "name": name,
            "phase": "start",
            "message": message,
        })
        rescue_plan = orchestrator.build_rescue_plan(attempt, origin_plan, numeric_qis)
        candidates = []
        if attempt == 1:
            rescue_plan = generator.validate_and_repair_plan(
                rescue_plan, real_train, cols["quasi_identifiers"]
            )
            # Search several deterministic seeds and DCR cutoffs. Generate a
            # surplus first so privacy pruning still returns the requested row
            # count. Each candidate must independently pass the full gate.
            for percentile in (35.0, 50.0, 65.0, 75.0, 85.0):
                for variant in range(3):
                    seed = 9400 + int(percentile) * 10 + variant
                    surplus_rows = max(
                        n_rows,
                        int(np.ceil(n_rows / (1.0 - percentile / 100.0))) + 5,
                    )
                    candidate = generator.synthesize(
                        real_train, rescue_plan, surplus_rows, cols["quasi_identifiers"],
                        label_column=cols["label"], generation_order=generation_order, seed=seed,
                    )
                    candidate = generator.drop_low_dcr_rows(
                        candidate, real_train, cols["quasi_identifiers"], percentile=percentile,
                    )
                    if len(candidate) > n_rows:
                        candidate = candidate.sample(n=n_rows, random_state=seed).reset_index(drop=True)
                    candidate = generator.suppress_unique_qi_combinations(
                        candidate, cols["quasi_identifiers"],
                    )
                    candidates.append((
                        candidate,
                        f"seed {seed}, DCR cutoff {percentile:.0f}%, singleton QIs suppressed",
                        rescue_plan,
                    ))
        elif attempt == 2:
            candidate = generator.generalize_frame(
                best_synth, real_train, cols["quasi_identifiers"], cols["label"]
            )
            candidates.append((candidate, "targeted QI generalization", rescue_plan))
        else:
            rescue_plan = generator.validate_and_repair_plan(
                rescue_plan, real_train, cols["quasi_identifiers"]
            )
            for variant in range(4):
                seed = 9303 + variant
                candidate = generator.second_generation(
                    best_synth, rescue_plan, n_rows, cols["quasi_identifiers"],
                    label_column=cols["label"], generation_order=generation_order, seed=seed,
                )
                candidates.append((candidate, f"second-generation seed {seed}", rescue_plan))

        # Pick the first fully passing candidate. If none pass, retain the
        # highest deterministic outcome score as the input to the next step.
        evaluated = []
        for candidate, variant_detail, candidate_plan in candidates:
            candidate_utility = statistician.evaluate(
                real_train, real_test, candidate, cols["label"], profile=profile,
            )
            candidate_attack = attacker.attack(
                profile, real_train, real_test, candidate, cols["quasi_identifiers"], prior=prior,
            )
            candidate_passed = orchestrator.decide(candidate_utility, candidate_attack, thresholds)
            candidate_score = orchestrator.outcome_score(candidate_utility, candidate_attack, thresholds)
            evaluated.append((
                candidate_score, candidate_passed, candidate, candidate_utility,
                candidate_attack, variant_detail, candidate_plan,
            ))
            if candidate_passed:
                break

        score, passed, current, utility, attack, variant_detail, rescue_plan = max(
            evaluated, key=lambda item: (item[1], item[0]),
        )
        detail = (
            f"{message} Selected {variant_detail} after evaluating {len(evaluated)} "
            f"deterministic candidate{'s' if len(evaluated) != 1 else ''}; kept {len(current)} rows."
        )
        record = RescueAttempt(
            attempt=attempt,
            name=name,
            passed=passed,
            utility=utility,
            attack=attack,
            plan=rescue_plan,
            rows_kept=int(len(current)),
            detail=detail,
        )
        attempts.append(record)
        round_result = RoundResult(
            round_index=max_rounds + attempt,
            utility=utility,
            attack=attack,
            passed=passed,
            plan=rescue_plan,
            orchestrator_feedback=detail,
            evaluated_candidates=[
                orchestrator.score_candidate(
                    0, rescue_plan, utility, attack, thresholds, max_rounds + attempt,
                ).model_copy(update={"composite_score": score}),
            ],
            winning_candidate_index=0,
        )
        extra_rounds.append(round_result)
        emit({
            "type": "rescue_attempt",
            "attempt": attempt,
            "name": name,
            "phase": "done",
            "passed": passed,
            "message": f"{'PASS' if passed else 'FAIL'} — {detail}",
            "rows_kept": int(len(current)),
            "result": round_result.model_dump(),
        })
        prior = {
            "re_id_rate": attack.re_id_rate,
            "membership_accuracy": attack.membership_accuracy,
            "generator_feedback": detail,
        }
        if passed:
            best_synth = current
            break
        if score >= best_score:
            best_score = score
            best_synth = current.copy()

    final_status = _certification(passed, rescued=passed)
    emit({
        "type": "rescue_finished",
        "final_status": final_status,
        "attempts_run": len(attempts),
        "passed": passed,
    })
    return {
        "synth": current,
        "passed": passed,
        "rounds": extra_rounds,
        "attempts": attempts,
    }


def run_forge(
    req: ForgeRequest,
    progress: Optional[ProgressFn] = None,
    run_id: Optional[str] = None,
) -> ForgeResponse:
    llm.reset_llm_state()
    settings = get_settings()
    if req.policy_source == "gemini":
        llm.record(
            "policy",
            "Translate the policy sentence into release-gate numbers.",
            settings.orchestrator_model,
            True,
        )
    run_id = run_id or uuid.uuid4().hex[:12]
    with _RUN_LOCK:
        _RUN_STATES.setdefault(run_id, {"status": "RUNNING", "progress": []})
    thresholds = req.thresholds or Thresholds(
        fidelity_min=settings.fidelity_min,
        tstr_min=settings.tstr_min,
        query_equivalence_min=settings.query_equivalence_min,
        reid_max=settings.reid_max,
        membership_max=settings.membership_max,
    )
    max_rounds = min(settings.max_rounds, thresholds.max_rounds, 5)

    def emit(event: dict):
        _record_event(run_id, event)
        if progress:
            progress(event)

    ai_seen = 0

    def flush_ai() -> None:
        nonlocal ai_seen
        log = llm.get_run_log()
        for item in log[ai_seen:]:
            emit({"type": "ai_usage", **item})
        ai_seen = len(log)

    # --- Ingest ---
    source = get_data_source(req.source)
    meta = source.meta(req.source)
    df = source.load(req.source)
    cols = _resolve_columns(req, meta)
    n_rows = req.n_rows or settings.default_n_rows
    difficulty = _assess_difficulty(len(df), cols["quasi_identifiers"])

    emit({
        "type": "loaded", "run_id": run_id, "rows": len(df), "columns": list(df.columns),
        "domain": meta.domain, "llm_enabled": llm.is_enabled(),
        "quasi_identifiers": cols["quasi_identifiers"],
        "sensitive_columns": cols["sensitive_columns"],
        "label_column": cols["label"], "max_rounds": max_rounds,
        "thresholds": thresholds.model_dump(),
        "auto_heal_enabled": req.auto_heal_enabled,
        "synthesis_mode": req.synthesis_mode,
        "dp_epsilon": req.dp_epsilon,
        **difficulty,
    })

    # --- Split IN (train/profile) vs OUT (held-out for TSTR + membership) ---
    stratify = None
    if cols["label"] and cols["label"] in df.columns:
        if df[cols["label"]].nunique() <= 20 and df[cols["label"]].value_counts().min() >= 2:
            stratify = df[cols["label"]]
    real_train, real_test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=stratify
    )
    real_train = real_train.reset_index(drop=True)
    real_test = real_test.reset_index(drop=True)

    # --- Profile (code; raw rows never leave this boundary) ---
    profile = profile_dataframe(
        real_train,
        quasi_identifiers=cols["quasi_identifiers"],
        sensitive_columns=cols["sensitive_columns"],
        label_column=cols["label"],
    )

    # --- LLM Profile Analyst ---
    emit({"type": "agent", "agent": "profiler", "message": "Analyst is inferring semantic dependencies and constraints..."})
    semantic_profile = profiler.analyze_profile(profile, req.business_rules)
    profile.semantic_profile = semantic_profile
    flush_ai()
    emit({"type": "profiled", "profile": profile.model_dump()})

    # Initialize Run Blackboard
    blackboard = RunBlackboard(run_id=run_id, semantic_profile=semantic_profile)

    # Initial plan
    initial_plan = generator.initial_plan(profile, req.business_rules)
    emit({"type": "plan", "round": 0, "plan": initial_plan.model_dump()})

    rounds: List[RoundResult] = []
    passed = False
    synth = None
    best_failed_synth = None
    best_failed_plan = None
    best_failed_score = float("-inf")
    passing_frames: Dict[str, pd.DataFrame] = {}
    released_id = ""
    prior_for_attacker: Optional[dict] = None
    dp_model = None
    if req.synthesis_mode == "dp":
        emit({
            "type": "agent", "agent": "generator", "done": False,
            "message": "Fitting Laplace noisy marginals under the epsilon budget...",
        })
        dp_model = dp_engine.fit_marginals(
            real_train,
            epsilon_total=req.dp_epsilon,
            label_column=cols["label"],
            seed=42,
        )
        emit({
            "type": "dp_budget",
            "epsilon_total": dp_model.epsilon_total,
            "epsilon_spent": dp_model.epsilon_spent,
            "queries": dp_model.query_names,
            "mechanism": "laplace",
        })
        shown = ", ".join(dp_model.query_names[:8])
        llm.advise(
            "generator",
            "Explain how the Laplace epsilon budget was split.",
            (
                "Explain this Laplace release in two sentences. No rows are included. "
                f"Epsilon spent {dp_model.epsilon_spent:.4f} of {dp_model.epsilon_total:.4f}. "
                f"Queries: {shown}. Later rounds redraw from this same noisy release."
            ),
            model=settings.generator_model,
        )
        flush_ai()
        emit({
            "type": "agent", "agent": "generator", "done": True,
            "message": (
                f"Laplace marginals released. Epsilon spent {dp_model.epsilon_spent:.4f} "
                f"of {dp_model.epsilon_total:.4f}."
            ),
        })

    for i in range(1, max_rounds + 1):
        if _is_aborted(run_id):
            logger.info("Run %s aborted before round %d start", run_id, i)
            _RUN_CACHE.pop(run_id, None)
            aborted_resp = ForgeResponse(
                run_id=run_id, status="ABORTED", dataset_key=req.source.dataset_key,
                domain=meta.domain, rounds=rounds, thresholds=thresholds,
                llm_enabled=llm.is_enabled(), difficulty=difficulty["difficulty"],
                difficulty_detail=difficulty["difficulty_detail"],
            )
            emit({"type": "done", "response": aborted_resp.model_dump()})
            return aborted_resp

        order = semantic_profile.conditional_generation_order if semantic_profile else None

        # Generate N candidate plans for this round. A pure DP run samples only
        # from the already released noisy marginals (post-processing).
        emit({
            "type": "agent", "agent": "generator", "round": i, "done": False,
            "message": (
                "Forge is preparing the next Laplace redraw..."
                if dp_model is not None
                else f"Forge is planning the Round {i} candidate beam..."
            ),
        })
        if dp_model is not None:
            candidates = [dp_engine.release_plan(dp_model)]
        elif i == 1:
            # Round 1 includes the naive baseline to establish pre-defense duel proof
            candidates = [initial_plan] + generator.generate_candidate_plans(
                profile, real_train, round_num=1, blackboard=blackboard, business_rules=req.business_rules
            )[:2]
        else:
            candidates = generator.generate_candidate_plans(
                profile, real_train, round_num=i, blackboard=blackboard, business_rules=req.business_rules
            )

        emit({"type": "round_start", "round": i, "plan": candidates[0].model_dump()})

        # 1. Forge (Generator) synthesizes candidates
        emit({
            "type": "agent", "agent": "generator", "round": i, "done": False,
            "message": (
                "Forge is sampling from the Laplace noisy marginals..."
                if dp_model is not None
                else f"Forge is exploring {len(candidates)} candidate plans along the Pareto frontier..."
            ),
        })

        candidate_synths: List[pd.DataFrame] = []
        if dp_model is not None:
            if not _is_aborted(run_id):
                candidate_synths.append(dp_engine.sample(dp_model, n_rows, seed=4200 + i))
        else:
            for c_idx, cand_plan in enumerate(candidates):
                if _is_aborted(run_id):
                    break
                cand_synth = generator.synthesize(
                    real_train, cand_plan, n_rows, cols["quasi_identifiers"],
                    label_column=cols["label"], generation_order=order, seed=42 + i * 10 + c_idx,
                )
                candidate_synths.append(cand_synth)

        emit({
            "type": "agent", "agent": "generator", "round": i, "done": True,
            "message": f"Synthesized {len(candidates)} Pareto candidates.",
        })

        if _is_aborted(run_id):
            logger.info("Run %s aborted during round %d candidate synthesis", run_id, i)
            _RUN_CACHE.pop(run_id, None)
            aborted_resp = ForgeResponse(
                run_id=run_id, status="ABORTED", dataset_key=req.source.dataset_key,
                domain=meta.domain, rounds=rounds, thresholds=thresholds,
                llm_enabled=llm.is_enabled(), difficulty=difficulty["difficulty"],
                difficulty_detail=difficulty["difficulty_detail"],
            )
            emit({"type": "done", "response": aborted_resp.model_dump()})
            return aborted_resp

        # 2. Auditor (Statistician) evaluates utility and diagnostics
        emit({
            "type": "agent", "agent": "statistician", "round": i, "done": False,
            "message": "Auditor is measuring fidelity, TSTR utility, and correlation matrices...",
        })

        cand_utils = []
        if not _is_aborted(run_id):
            # Beam candidates are independent within a round. Evaluate them
            # concurrently while preserving their original list order.
            with ThreadPoolExecutor(
                max_workers=min(3, len(candidate_synths)),
                thread_name_prefix="auditor",
            ) as pool:
                cand_utils = list(pool.map(
                    lambda frame: statistician.evaluate(
                        real_train, real_test, frame, cols["label"], profile=profile,
                    ),
                    candidate_synths,
                ))

        best_fid = max(u.fidelity_score for u in cand_utils) if cand_utils else 0.0
        best_tstr = max(u.utility_ratio for u in cand_utils) if cand_utils else 0.0
        emit({
            "type": "agent", "agent": "statistician", "round": i, "done": True,
            "message": f"Utility {int(best_tstr * 100)}% | fidelity {int(best_fid * 100)}%.",
        })

        if _is_aborted(run_id):
            logger.info("Run %s aborted during round %d utility evaluation", run_id, i)
            _RUN_CACHE.pop(run_id, None)
            aborted_resp = ForgeResponse(
                run_id=run_id, status="ABORTED", dataset_key=req.source.dataset_key,
                domain=meta.domain, rounds=rounds, thresholds=thresholds,
                llm_enabled=llm.is_enabled(), difficulty=difficulty["difficulty"],
                difficulty_detail=difficulty["difficulty_detail"],
            )
            emit({"type": "done", "response": aborted_resp.model_dump()})
            return aborted_resp

        # 3. Challenger (Attacker) launches adaptive privacy attacks
        emit({
            "type": "agent", "agent": "attacker", "round": i, "done": False,
            "message": "Challenger is launching adaptive re-identification and membership attacks...",
        })

        cand_attacks = []
        if not _is_aborted(run_id):
            # The initial attack prompt depends on the round/profile, not on a
            # candidate. Ask Gemini once, then run each candidate's deterministic
            # tools and adaptive follow-up concurrently.
            shared_attack_plan = attacker.plan_attack(
                profile, cols["quasi_identifiers"], prior_for_attacker,
            )
            with ThreadPoolExecutor(
                max_workers=min(3, len(candidate_synths)),
                thread_name_prefix="challenger",
            ) as pool:
                cand_attacks = list(pool.map(
                    lambda frame: attacker.attack(
                        profile,
                        real_train,
                        real_test,
                        frame,
                        cols["quasi_identifiers"],
                        prior=prior_for_attacker,
                        initial_plan=shared_attack_plan,
                    ),
                    candidate_synths,
                ))

        emit({
            "type": "agent", "agent": "attacker", "round": i, "done": True,
            "message": cand_attacks[0].attacker_strategy if cand_attacks else "Attack complete.",
        })

        if _is_aborted(run_id):
            logger.info("Run %s aborted during round %d attack evaluation", run_id, i)
            _RUN_CACHE.pop(run_id, None)
            aborted_resp = ForgeResponse(
                run_id=run_id, status="ABORTED", dataset_key=req.source.dataset_key,
                domain=meta.domain, rounds=rounds, thresholds=thresholds,
                llm_enabled=llm.is_enabled(), difficulty=difficulty["difficulty"],
                difficulty_detail=difficulty["difficulty_detail"],
            )
            emit({"type": "done", "response": aborted_resp.model_dump()})
            return aborted_resp

        # 4. Arbiter (Orchestrator) evaluates Pareto frontier and certifies release gates
        emit({
            "type": "agent", "agent": "orchestrator", "round": i, "done": False,
            "message": "Arbiter is verifying release thresholds against Pareto frontier...",
        })
        evaluations: List[CandidateEvaluation] = []
        for c_idx in range(len(candidates)):
            cand_plan = candidates[c_idx]
            cand_util = cand_utils[c_idx]
            cand_attack = cand_attacks[c_idx]
            evaluation = orchestrator.score_candidate(
                c_idx, cand_plan, cand_util, cand_attack, thresholds, i,
            )
            evaluations.append(evaluation)
            if evaluation.gate_passed and c_idx < len(candidate_synths):
                passing_frames[evaluation.candidate_id] = candidate_synths[c_idx].copy()

        # Deterministic Pareto candidate selection
        best_idx = orchestrator.select_pareto_candidate(evaluations, thresholds)
        winning_eval = evaluations[best_idx]
        synth = candidate_synths[best_idx]
        winning_plan = winning_eval.plan
        utility = winning_eval.utility
        attack = winning_eval.attack
        passed = orchestrator.decide(utility, attack, thresholds)

        feedback = ""
        if not passed and i < max_rounds:
            feedback = orchestrator.diagnose_and_advise(blackboard, thresholds)
            blackboard.orchestrator_feedback_history.append(feedback)
            emit({
                "type": "agent", "agent": "orchestrator", "round": i, "done": True,
                "message": feedback,
            })
        else:
            arb_msg = (
                f"CERTIFIED in Round {i}: Candidate '{winning_plan.plan_name}' satisfies all release thresholds!"
                if passed
                else "Evaluated round thresholds."
            )
            emit({
                "type": "agent", "agent": "orchestrator", "round": i, "done": True,
                "message": arb_msg,
            })

        round_result = RoundResult(
            round_index=i,
            utility=utility,
            attack=attack,
            passed=passed,
            plan=winning_plan,
            orchestrator_feedback=feedback,
            evaluated_candidates=evaluations,
            winning_candidate_index=best_idx,
        )
        rounds.append(round_result)
        blackboard.round_history.append(round_result)

        if utility.diagnostic_report:
            for w in utility.diagnostic_report.weakest_columns:
                if w not in blackboard.weakest_columns_history:
                    blackboard.weakest_columns_history.append(w)
            blackboard.broken_relationships_history.extend(utility.diagnostic_report.broken_relationships[:3])

        emit({"type": "round_result", "round": i, "result": round_result.model_dump()})

        if not passed and synth is not None and winning_eval.composite_score >= best_failed_score:
            best_failed_score = winning_eval.composite_score
            best_failed_synth = synth.copy()
            best_failed_plan = winning_plan

        if passed:
            released_id = winning_eval.candidate_id
            break

        prior_for_attacker = {
            "re_id_rate": attack.re_id_rate,
            "membership_accuracy": attack.membership_accuracy,
            "generator_feedback": feedback,
        }
        flush_ai()

    flush_ai()

    if _is_aborted(run_id):
        _RUN_CACHE.pop(run_id, None)
        aborted_resp = ForgeResponse(
            run_id=run_id, status="ABORTED", dataset_key=req.source.dataset_key,
            domain=meta.domain, rounds=rounds, thresholds=thresholds,
            llm_enabled=llm.is_enabled(), difficulty=difficulty["difficulty"],
            difficulty_detail=difficulty["difficulty_detail"],
        )
        emit({"type": "done", "response": aborted_resp.model_dump()})
        return aborted_resp

    rescue_attempts: List[RescueAttempt] = []
    if (
        not passed
        and req.auto_heal_enabled
        and best_failed_synth is not None
        and best_failed_plan is not None
        and not _is_aborted(run_id)
    ):
        healed = _auto_heal(
            emit=emit,
            run_id=run_id,
            profile=profile,
            real_train=real_train,
            real_test=real_test,
            synth=best_failed_synth,
            plan=best_failed_plan,
            cols=cols,
            n_rows=n_rows,
            thresholds=thresholds,
            generation_order=order,
            prior=prior_for_attacker,
            max_rounds=max_rounds,
            best_score=best_failed_score,
            failure_brief=(
                f"fidelity {utility.fidelity_score}, prediction {utility.utility_ratio}, "
                f"query {utility.query_equivalence}, re-identification {attack.re_id_rate}, "
                f"membership {attack.membership_accuracy}, attribute inference {attack.attribute_inference_advantage}, "
                f"linkage {attack.linkage_rate}, near-duplicate {attack.near_duplicate_rate}."
            ),
        )
        flush_ai()
        if healed is None:
            _RUN_CACHE.pop(run_id, None)
            aborted_resp = ForgeResponse(
                run_id=run_id, status="ABORTED", dataset_key=req.source.dataset_key,
                domain=meta.domain, rounds=rounds, thresholds=thresholds,
                llm_enabled=llm.is_enabled(), difficulty=difficulty["difficulty"],
                difficulty_detail=difficulty["difficulty_detail"],
                final_status="NOT_CERTIFIED",
                auto_heal_enabled=req.auto_heal_enabled,
            )
            emit({"type": "done", "response": aborted_resp.model_dump()})
            return aborted_resp
        synth = healed["synth"]
        passed = healed["passed"]
        rounds.extend(healed["rounds"])
        rescue_attempts = healed["attempts"]
        for rescued_round in healed["rounds"]:
            blackboard.round_history.append(rescued_round)
        if passed and healed["rounds"]:
            rescue_eval = healed["rounds"][-1].evaluated_candidates[0]
            released_id = rescue_eval.candidate_id
            passing_frames = {released_id: synth.copy()}

    final_status = _certification(passed, rescued=bool(rescue_attempts) and passed)
    status = "SUCCESS" if passed else "FAILED_TO_CONVERGE"
    frontier = [candidate for rnd in rounds for candidate in rnd.evaluated_candidates]
    orchestrator.mark_pareto(frontier)
    if passed and frontier and not released_id:
        winner = rounds[-1]
        released_id = f"r{winner.round_index}-c{winner.winning_candidate_index}"
    with _RUN_LOCK:
        if passed and passing_frames:
            _PASSING_FRAMES[run_id] = passing_frames
        else:
            _PASSING_FRAMES.pop(run_id, None)

    # --- Export + scorecard ---
    export_path = None
    final_table_id = None
    if synth is not None and not _is_aborted(run_id):
        _RUN_CACHE[run_id] = synth
        export_path = reporter.export_csv(synth, run_id)
        if passed and req.destination_table_id:
            final_table_id = reporter.publish_bigquery(synth, req.destination_table_id)

    fairness_report = (
        rounds[-1].utility.fairness
        if rounds and rounds[-1].utility.fairness and rounds[-1].utility.fairness.summary
        else None
    )
    # Reporter and fairness prose are independent advisory calls. Running them
    # together avoids two serial Vertex round trips before the final UI update.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="reporter") as pool:
        scorecard_future = pool.submit(
            reporter.build_scorecard,
            rounds,
            status,
            thresholds,
            meta.domain,
            difficulty["difficulty"],
            difficulty["difficulty_detail"],
        )
        fairness_future = None
        if fairness_report is not None:
            fairness_future = pool.submit(
                llm.advise,
                "statistician",
                "Explain the fairness findings without changing the release decision.",
                (
                    "Fairness is advisory and does not block certification. "
                    f"Status: {fairness_report.status}. "
                    f"Summary: {fairness_report.summary}"
                ),
                model=settings.reporter_model,
            )
        scorecard = scorecard_future.result()
        fairness_note = fairness_future.result() if fairness_future is not None else ""
    if final_status == "CERTIFIED_AFTER_RESCUE":
        scorecard += (
            "\n\nAUTO-HEAL: CERTIFIED_AFTER_RESCUE. A rescue attempt passed the "
            "release gate. Metrics were recomputed on the original real split."
        )
    elif rescue_attempts:
        scorecard += (
            "\n\nAUTO-HEAL: NOT_CERTIFIED. The rescue ladder did not pass the release gate."
        )
    pure_dp = dp_model is not None and not rescue_attempts
    formal_dp_guarantee = dp_engine.formal_dp_guarantee_text(
        pure_dp, dp_model.epsilon_spent if dp_model is not None else 0.0
    )
    scorecard += f"\n\nDIFFERENTIAL PRIVACY: {formal_dp_guarantee}"
    if fairness_note:
        scorecard += f"\n\nFAIRNESS NARRATIVE: {fairness_note}"
    if fairness_report is not None:
        flush_ai()
    if frontier:
        optimal = sum(1 for candidate in frontier if candidate.is_pareto_optimal)
        certified_count = sum(1 for candidate in frontier if candidate.gate_passed)
        scorecard += (
            f"\n\nPARETO FRONTIER: {optimal} of {len(frontier)} candidates are non-dominated. "
            f"{certified_count} passed the release gate."
        )
    dual_scorecard = reporter.build_dual_scorecard(rounds, status, thresholds)
    col_dist = None
    if synth is not None and real_train is not None and not _is_aborted(run_id):
        try:
            col_dist = compute_column_distributions(real_train, synth)
        except Exception as exc:
            logger.warning("Could not compute column distributions for run %s: %s", run_id, exc)

    response = ForgeResponse(
        run_id=run_id,
        status=status,
        dataset_key=req.source.dataset_key,
        domain=meta.domain,
        rounds=rounds,
        final_table_id=final_table_id,
        export_path=export_path,
        scorecard_summary=scorecard,
        scorecard=dual_scorecard,
        thresholds=thresholds,
        llm_enabled=llm.is_enabled(),
        difficulty=difficulty["difficulty"],
        difficulty_detail=difficulty["difficulty_detail"],
        final_status=final_status,
        rescue_attempts=rescue_attempts,
        auto_heal_enabled=req.auto_heal_enabled,
        formal_dp_guarantee=formal_dp_guarantee,
        active_candidate_id=released_id if passed else "",
        policy_name=req.policy_name,
        policy_source=req.policy_source,
        policy_explanation=req.policy_explanation,
        column_distributions=col_dist,
    )
    if passed and synth is not None:
        try:
            certificate.issue(run_id, response.model_dump(), synth, real_train)
        except Exception:
            logger.exception("Could not issue the release certificate for %s", run_id)
    elif not passed:
        try:
            failure_report.issue(
                run_id,
                response.model_dump(),
                {
                    "synthesis_mode": req.synthesis_mode,
                    "dp_epsilon": req.dp_epsilon,
                    "epsilon_spent": dp_model.epsilon_spent if dp_model is not None else None,
                    "dp_queries": list(dp_model.query_names) if dp_model is not None else [],
                    "n_rows": n_rows,
                    # The scorecard already contains the Reporter model's
                    # executive summary. Avoid another serial LLM wait here.
                    "skip_ai_narrative": True,
                },
            )
        except Exception:
            logger.exception("Could not issue the failure report for %s", run_id)
    flush_ai()
    response.ai_usage = [AIUsageStep.model_validate(item) for item in llm.get_run_log()]
    emit({"type": "done", "response": response.model_dump()})
    return response
