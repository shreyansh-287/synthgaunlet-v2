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
from typing import Callable, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

import llm
import reporter
from agents import attacker, generator, orchestrator, profiler, statistician
from agents.profiler import profile_dataframe
from config import get_settings
from data_source import get_data_source
from models import (
    CandidateEvaluation,
    ForgeRequest,
    ForgeResponse,
    RoundResult,
    RunBlackboard,
    RunStartResponse,
    RunStatusResponse,
    Thresholds,
)

logger = logging.getLogger("forge.pipeline")

# In-memory registry of the latest synthetic frames, keyed by run_id (for export).
_RUN_CACHE: Dict[str, pd.DataFrame] = {}
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


def run_forge(
    req: ForgeRequest,
    progress: Optional[ProgressFn] = None,
    run_id: Optional[str] = None,
) -> ForgeResponse:
    llm.reset_llm_state()
    settings = get_settings()
    run_id = run_id or uuid.uuid4().hex[:12]
    with _RUN_LOCK:
        _RUN_STATES.setdefault(run_id, {"status": "RUNNING", "progress": []})
    max_rounds = min(settings.max_rounds, 5)
    thresholds = req.thresholds or Thresholds(
        fidelity_min=settings.fidelity_min,
        tstr_min=settings.tstr_min,
        query_equivalence_min=settings.query_equivalence_min,
        reid_max=settings.reid_max,
        membership_max=settings.membership_max,
    )

    def emit(event: dict):
        _record_event(run_id, event)
        if progress:
            progress(event)

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
        "thresholds": thresholds.model_dump(), **difficulty,
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
    emit({"type": "profiled", "profile": profile.model_dump()})

    # Initialize Run Blackboard
    blackboard = RunBlackboard(run_id=run_id, semantic_profile=semantic_profile)

    # Initial plan
    initial_plan = generator.initial_plan(profile, req.business_rules)
    emit({"type": "plan", "round": 0, "plan": initial_plan.model_dump()})

    rounds: List[RoundResult] = []
    passed = False
    synth = None
    prior_for_attacker: Optional[dict] = None

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

        # Generate N candidate plans for this round
        if i == 1:
            # Round 1 includes the naive baseline to establish pre-defense duel proof
            candidates = [initial_plan] + generator.generate_candidate_plans(
                profile, real_train, round_num=1, blackboard=blackboard, business_rules=req.business_rules
            )[:2]
        else:
            candidates = generator.generate_candidate_plans(
                profile, real_train, round_num=i, blackboard=blackboard, business_rules=req.business_rules
            )

        emit({"type": "round_start", "round": i, "plan": candidates[0].model_dump()})

        order = semantic_profile.conditional_generation_order if semantic_profile else None

        # 1. Forge (Generator) synthesizes candidates
        emit({
            "type": "agent", "agent": "generator", "round": i, "done": False,
            "message": f"Forge is exploring {len(candidates)} candidate plans along the Pareto frontier...",
        })

        candidate_synths: List[pd.DataFrame] = []
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
        for cand_synth in candidate_synths:
            if _is_aborted(run_id):
                break
            cand_utils.append(statistician.evaluate(real_train, real_test, cand_synth, cols["label"]))

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
        for cand_synth in candidate_synths:
            if _is_aborted(run_id):
                break
            cand_attacks.append(
                attacker.attack(
                    profile, real_train, real_test, cand_synth, cols["quasi_identifiers"],
                    prior=prior_for_attacker,
                )
            )

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
            cand_passed = orchestrator.decide(cand_util, cand_attack, thresholds)
            evaluations.append(
                CandidateEvaluation(
                    candidate_index=c_idx,
                    plan=cand_plan,
                    utility=cand_util,
                    attack=cand_attack,
                    passed=cand_passed,
                )
            )

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

        if passed:
            break

        prior_for_attacker = {
            "re_id_rate": attack.re_id_rate,
            "membership_accuracy": attack.membership_accuracy,
            "generator_feedback": feedback,
        }

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

    status = "SUCCESS" if passed else "FAILED_TO_CONVERGE"

    # --- Export + scorecard ---
    export_path = None
    final_table_id = None
    if synth is not None and not _is_aborted(run_id):
        _RUN_CACHE[run_id] = synth
        export_path = reporter.export_csv(synth, run_id)
        if passed and req.destination_table_id:
            final_table_id = reporter.publish_bigquery(synth, req.destination_table_id)

    scorecard = reporter.build_scorecard(
        rounds, status, thresholds, meta.domain,
        difficulty=difficulty["difficulty"],
        difficulty_detail=difficulty["difficulty_detail"],
    )
    dual_scorecard = reporter.build_dual_scorecard(rounds, status, thresholds)

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
    )
    emit({"type": "done", "response": response.model_dump()})
    return response
