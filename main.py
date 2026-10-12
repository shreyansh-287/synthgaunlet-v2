"""FastAPI application for Synthetic Data Forge.

Runs locally (uvicorn) and on Cloud Run (same container). Endpoints:
  GET  /health                     - liveness probe
  GET  /api/v1/datasets            - static dataset catalog
  POST /api/v1/forge               - run the adversarial loop (sync)
  GET  /api/v1/forge/stream        - run with live SSE progress
  GET  /api/v1/export/{run_id}     - download synthetic CSV
  GET  /api/v1/bq/datasets         - list BigQuery datasets
  GET  /api/v1/bq/tables           - list BigQuery tables in a dataset
  POST /api/v1/bq/import           - import a BigQuery table as upload
  GET  /                           - demo UI
"""

from __future__ import annotations

import io
import json
import logging
import math
import queue
import re
import threading
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import JSONResponse

import certificate
import llm
import bigquery_client as bq
import pii_scanner
from ingest import IngestError, load_tabular
from agents import policy
from agents.adk_agents import adk_available
from history_store import HistoryItem, list_history, save_history, clear_history, delete_history_item
from config import get_settings
from data_source import get_data_source, list_datasets, register_upload, replace_upload
from models import ForgeRequest, ForgeResponse, RunStartResponse, RunStatusResponse, SourceSpec, Thresholds
from pipeline import (
    abort_run,
    activate_passing_candidate,
    get_passing_frame,
    get_run_frame,
    get_run_status,
    run_forge,
    start_run,
)

logger = logging.getLogger("forge.main")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB
MAX_UPLOAD_ROWS = 50_000              # keep the adversarial loop responsive
MIN_UPLOAD_ROWS = 15                  # accessible for small sample evaluation


def _split_csv_param(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replaces NaN, Infinity, -Infinity and pd.NA with None so Starlette JSON dumps never fail."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj


class SafeJSONResponse(JSONResponse):
    """Drop-in JSONResponse that guarantees no NaN/Inf causes a 500 response error."""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            _sanitize_for_json(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


def _safe_preview(df: pd.DataFrame, n: int = 5) -> List[dict]:
    """Return the first n rows as string-clean dicts with NaN/nulls replaced with empty strings."""
    records = []
    for row in df.head(n).to_dict(orient="records"):
        cleaned = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                cleaned[str(k)] = ""
            elif str(v) in ("nan", "NaN", "None", "<NA>"):
                cleaned[str(k)] = ""
            else:
                cleaned[str(k)] = str(v)
        records.append(cleaned)
    return records


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="SynthGauntlet", version="2.0.0", default_response_class=SafeJSONResponse)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "synthgauntlet",
        "version": "2.0.0",
        "llm_enabled": llm.is_enabled(),
        "llm_provider": llm.provider_name(),
        "agent_framework": "google-adk" if adk_available() else "deterministic-compatible",
        "tokens": llm.get_token_usage(),
    }


@app.get("/api/v1/tokens")
def tokens():
    return llm.get_token_usage()


@app.get("/api/v1/history")
@app.get("/history")
def history():
    items, persistent = list_history()
    return {"history": items, "persistent": persistent}


@app.post("/api/v1/history")
@app.post("/history")
def write_history(item: HistoryItem):
    saved, persistent = save_history(item)
    return {"item": saved, "persistent": persistent}


@app.delete("/api/v1/history")
def delete_all_history():
    cleared = clear_history()
    return {"cleared": cleared}


@app.delete("/api/v1/history/{item_id}")
def delete_single_history_item(item_id: str):
    deleted = delete_history_item(item_id)
    return {"deleted": deleted, "id": item_id}



# ---------------------------------------------------------------------------
# BigQuery endpoints
# ---------------------------------------------------------------------------

def _extract_token(authorization: Optional[str] = None) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        tok = authorization.split(" ", 1)[1].strip()
        if tok and tok.lower() not in ("null", "undefined", "none", ""):
            return tok
    return None


@app.get("/api/v1/auth/config")
def auth_config():
    """Return public OAuth client configuration for client-side Google sign-in."""
    settings = get_settings()
    import os
    client_id = getattr(settings, "google_client_id", "") or os.getenv("GOOGLE_CLIENT_ID", "")
    return {"google_client_id": client_id}


@app.get("/api/v1/bq/datasets")
def bq_datasets(project: str = "", authorization: Optional[str] = Header(None)):
    """List all BigQuery dataset IDs in the given GCP project."""
    settings = get_settings()
    project = project or settings.gcp_project
    user_token = _extract_token(authorization)
    if not project:
        raise HTTPException(status_code=400, detail="GCP project ID is required. Set GCP_PROJECT in .env or pass ?project=...")
    if not user_token and not bq.is_bq_available():
        raise HTTPException(status_code=503, detail="BigQuery is not available. Connect with Google or run: gcloud auth application-default login")
    try:
        return {"project": project, "datasets": bq.list_bq_datasets(project, user_token=user_token)}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/v1/bq/tables")
def bq_tables(project: str = "", dataset: str = "", authorization: Optional[str] = Header(None)):
    """List all table IDs in a BigQuery dataset."""
    settings = get_settings()
    project = project or settings.gcp_project
    user_token = _extract_token(authorization)
    if not project:
        raise HTTPException(status_code=400, detail="GCP project ID is required.")
    if not dataset:
        raise HTTPException(status_code=400, detail="dataset query param is required.")
    if not user_token and not bq.is_bq_available():
        raise HTTPException(status_code=503, detail="BigQuery is not available.")
    try:
        return {"project": project, "dataset": dataset, "tables": bq.list_bq_tables(project, dataset, user_token=user_token)}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/v1/bq/table-info")
def bq_table_info(project: str = "", dataset: str = "", table: str = "", authorization: Optional[str] = Header(None)):
    """Return row count and metadata for a BigQuery table."""
    settings = get_settings()
    project = project or settings.gcp_project
    user_token = _extract_token(authorization)
    if not project:
        raise HTTPException(status_code=400, detail="GCP project ID is required.")
    if not dataset or not table:
        raise HTTPException(status_code=400, detail="dataset and table query params are required.")
    if not user_token and not bq.is_bq_available():
        raise HTTPException(status_code=503, detail="BigQuery is not available.")
    return bq.get_bq_table_info(project, dataset, table, user_token=user_token)


class BQImportRequest(BaseModel):
    project: str = ""
    dataset: str
    table: str
    limit: int = 10000
    user_token: Optional[str] = None


@app.post("/api/v1/bq/import")
def bq_import(body: BQImportRequest, authorization: Optional[str] = Header(None)):
    """Fetch a BigQuery table and register it as an upload (returns upload_id)."""
    settings = get_settings()
    project = body.project or settings.gcp_project
    user_token = body.user_token or _extract_token(authorization)
    if not project:
        raise HTTPException(status_code=400, detail="GCP project ID is required.")
    if not user_token and not bq.is_bq_available():
        raise HTTPException(status_code=503, detail="BigQuery is not available. Run: gcloud auth application-default login")

    limit = max(50, min(body.limit or 10000, 10000))

    try:
        table_meta = bq.get_bq_table_info(project, body.dataset, body.table, user_token=user_token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not retrieve table metadata: {exc}")

    try:
        df = bq.fetch_bq_table(project, body.dataset, body.table, limit=limit, user_token=user_token)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Unexpected error fetching table: {exc}")

    if len(df) < MIN_UPLOAD_ROWS or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail=f"Table too small — need at least {MIN_UPLOAD_ROWS} rows and 2 columns.")

    try:
        filename = f"{body.dataset}.{body.table}.bq"
        upload_id, clean_df, meta = register_upload(df, filename=filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process imported table: {exc}")

    return {
        "upload_id": upload_id,
        "source": "bigquery",
        "table": f"{project}.{body.dataset}.{body.table}",
        "filename": f"{body.dataset}.{body.table}",
        "rows": int(len(clean_df)),
        "total_table_rows": int(table_meta.get("total_rows") or len(clean_df)),
        "columns": [str(c) for c in clean_df.columns],
        "numeric_columns": [str(c) for c in clean_df.select_dtypes("number").columns],
        "quasi_identifiers": [str(c) for c in meta.get("quasi_identifiers", [])],
        "sensitive_columns": [str(c) for c in meta.get("sensitive_columns", [])],
        "label_column": str(meta["label_column"]) if meta.get("label_column") is not None else None,
        "preview": _safe_preview(clean_df, 5),
    }


class BQExportRequest(BaseModel):
    run_id: str
    project: str = ""
    dataset: str = ""
    table: str


class ActiveCandidateRequest(BaseModel):
    candidate_id: str


class PolicyRequest(BaseModel):
    request: str


@app.get("/api/v1/policy/presets")
def policy_presets():
    return {"presets": policy.presets_payload()}


@app.post("/api/v1/policy/interpret")
def interpret_policy(body: PolicyRequest):
    text = body.request.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Describe the policy you want.")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Policy description must be at most 2,000 characters.")
    return policy.interpret(text).model_dump()


@app.post("/api/v1/bq/export")
def bq_export(body: BQExportRequest):
    """Export a run's synthetic DataFrame directly into a Google BigQuery table."""
    settings = get_settings()
    project = body.project or settings.gcp_project
    if not project:
        raise HTTPException(status_code=400, detail="Target GCP project ID is required.")
    if not body.dataset:
        raise HTTPException(status_code=400, detail="Target BigQuery dataset is required.")
    if not body.table:
        raise HTTPException(status_code=400, detail="Target BigQuery table name is required.")

    blocked = _uncertified_download_detail(body.run_id)
    if blocked:
        raise HTTPException(status_code=403, detail=blocked)

    if not bq.is_bq_available():
        raise HTTPException(
            status_code=503,
            detail="BigQuery is not available. Please ensure Google Cloud Application Default Credentials are authenticated.",
        )

    # 1. Retrieve synthetic DataFrame for this run
    df = get_run_frame(body.run_id)
    if df is None or df.empty:
        # Fallback to saved CSV on disk if in-memory cache evicted
        csv_file = BASE_DIR / "data" / "output" / f"{body.run_id}.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
            except Exception as read_exc:
                logger.warning("Could not read csv for export: %s", read_exc)

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic data for run_id '{body.run_id}' not found. Please forge synthetic data first.",
        )

    try:
        res = bq.export_synthetic_to_bq(
            df=df,
            project=project,
            dataset=body.dataset,
            table=body.table,
        )
        return res
    except ValueError as val_exc:
        raise HTTPException(status_code=400, detail=str(val_exc))
    except RuntimeError as run_exc:
        raise HTTPException(status_code=502, detail=str(run_exc))
    except Exception as exc:
        logger.exception("Unexpected error during BigQuery export: %s", exc)
        raise HTTPException(status_code=500, detail=f"BigQuery export failed: {exc}")


@app.get("/api/v1/datasets")
def datasets():
    return {"datasets": list_datasets()}


@app.get("/api/v1/sample/profile")
def sample_profile(dataset_key: str):
    """Fetch column profiling info for a sample dataset."""
    try:
        spec = SourceSpec(type="csv", dataset_key=dataset_key)
        src = get_data_source(spec)
        df = src.load(spec)
        meta = src.meta(spec)
        return {
            "dataset_key": dataset_key,
            "filename": dataset_key,
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "numeric_columns": [str(c) for c in df.select_dtypes("number").columns],
            "quasi_identifiers": [str(c) for c in meta.get("quasi_identifiers", [])],
            "sensitive_columns": [str(c) for c in meta.get("sensitive_columns", [])],
            "label_column": str(meta["label_column"]) if meta.get("label_column") is not None else None,
            "preview": _safe_preview(df, 5),
        }
    except Exception as exc:
        logger.exception("sample_profile error for %s: %s", dataset_key, exc)
        raise HTTPException(status_code=404, detail=f"Sample dataset '{dataset_key}' not found: {exc}")


# ---------------------------------------------------------------------------
# Run result persistence — save & restore full ForgeResponse by run_id
# ---------------------------------------------------------------------------

RUN_RESULTS_DIR = BASE_DIR / "data" / "run_results"
RUN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_RUN_MAP_FILE = BASE_DIR / "data" / "history_run_map.json"


def _load_run_map() -> dict:
    if HISTORY_RUN_MAP_FILE.exists():
        try:
            return json.loads(HISTORY_RUN_MAP_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_run_map(m: dict) -> None:
    try:
        HISTORY_RUN_MAP_FILE.write_text(json.dumps(m, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save history_run_map: %s", exc)


@app.post("/api/v1/runs/{run_id}")
def save_run_result(run_id: str, payload: dict):
    """Persist a full ForgeResponse JSON so history can restore it later."""
    try:
        out = RUN_RESULTS_DIR / f"{run_id}.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        history_id = payload.get("history_id")
        if history_id:
            m = _load_run_map()
            m[history_id] = run_id
            _save_run_map(m)
        return {"saved": True, "run_id": run_id}
    except Exception as exc:
        logger.error("save_run_result error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/runs/{run_id}")
def get_run_result(run_id: str):
    """Retrieve a previously saved ForgeResponse JSON."""
    path = RUN_RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run result '{run_id}' not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/history/{history_id}/run")
def get_run_for_history(history_id: str):
    """Look up and return the run result for a given history item id."""
    m = _load_run_map()
    run_id = m.get(history_id)
    if not run_id:
        raise HTTPException(status_code=404, detail="No run result saved for this history item")
    path = RUN_RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run result file missing for run_id '{run_id}'")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _upload_payload(upload_id: str, filename: Optional[str], clean_df: pd.DataFrame, meta: dict, warnings: List[str]) -> dict:
    findings = pii_scanner.scan(clean_df)
    preview = _safe_preview(clean_df, 5)
    sensitive = {item["column"] for item in findings}
    for row in preview:
        for column in sensitive:
            if row.get(column):
                row[column] = "[redacted]"
    return {
        "upload_id": upload_id,
        "filename": filename,
        "rows": int(len(clean_df)),
        "columns": [str(c) for c in clean_df.columns],
        "numeric_columns": [str(c) for c in clean_df.select_dtypes("number").columns],
        "quasi_identifiers": [str(c) for c in meta.get("quasi_identifiers", [])],
        "sensitive_columns": [str(c) for c in meta.get("sensitive_columns", [])],
        "label_column": str(meta["label_column"]) if meta.get("label_column") is not None else None,
        "preview": preview,
        "warnings": warnings,
        "pii_findings": findings,
    }


def _reject_small_upload(clean_df: pd.DataFrame) -> None:
    raw_rows = int(len(clean_df))
    raw_cols = int(clean_df.shape[1])
    if raw_rows < MIN_UPLOAD_ROWS or raw_cols < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has only {raw_rows} row(s) and {raw_cols} column(s). Need at least {MIN_UPLOAD_ROWS} rows and 2 columns for synthesis & evaluation.",
        )


@app.post("/upload")
@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None)):
    """Accept a table, clean it, and auto-detect column roles.

    Returns an upload_id plus the inferred label / quasi-identifiers so the UI
    can let the user confirm or adjust before forging."""
    if not isinstance(sheet_name, str) or not sheet_name.strip():
        sheet_name = None
    fname = (file.filename or "").lower()
    suffix = ".jsonl" if fname.endswith(".jsonl") else Path(fname).suffix
    if suffix in {".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 15 MB).")
        try:
            frame, warnings = load_tabular(raw, fname, sheet_name)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        upload_id, clean_df, meta = register_upload(frame, filename=file.filename or "upload.csv")
        _reject_small_upload(clean_df)
        return _upload_payload(upload_id, file.filename, clean_df, meta, warnings)
    valid_csv_content_types = (
        "text/csv",
        "text/plain",
        "application/vnd.ms-excel",
        "application/csv",
        "text/x-csv",
        "text/comma-separated-values",
        "application/octet-stream",
    )
    disallowed_extensions = (
        ".json", ".xlsx", ".xls", ".pdf", ".zip", ".tar", ".gz",
        ".png", ".jpg", ".jpeg", ".parquet", ".feather", ".h5", ".pkl"
    )
    is_csv = (
        fname.endswith(".csv")
        or fname.endswith(".csv.txt")
        or (
            (file.content_type in valid_csv_content_types or not file.content_type)
            and not fname.endswith(disallowed_extensions)
        )
    )
    if not is_csv:
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB).")

    try:
        # Multi-encoding & multi-delimiter auto-detection (comma, semicolon, tab, pipe)
        text = None
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")

        # Strip null bytes if present
        text = text.replace("\x00", "")

        df = None
        try:
            candidate = pd.read_csv(io.StringIO(text), sep=",", skipinitialspace=True)
            if candidate.shape[1] > 1:
                df = candidate
            else:
                df = candidate
        except Exception:
            df = None

        if df is None or df.shape[1] <= 1:
            for sep in (";", "\t", "|", None):
                try:
                    candidate = pd.read_csv(io.StringIO(text), sep=sep, engine="python", skipinitialspace=True)
                    if df is None or candidate.shape[1] > df.shape[1]:
                        df = candidate
                except Exception:
                    continue

        if df is None:
            raise ValueError("Could not parse file as CSV.")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if len(df) > MAX_UPLOAD_ROWS:
        df = df.sample(n=MAX_UPLOAD_ROWS, random_state=42).reset_index(drop=True)

    upload_id, clean_df, meta = register_upload(df, filename=file.filename or "upload.csv")

    raw_rows = int(len(clean_df))
    raw_cols = int(clean_df.shape[1])

    if raw_rows < MIN_UPLOAD_ROWS or raw_cols < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has only {raw_rows} row(s) and {raw_cols} column(s). Need at least {MIN_UPLOAD_ROWS} rows and 2 columns for synthesis & evaluation.",
        )
    if raw_rows < MIN_UPLOAD_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has only {raw_rows} row(s). Need at least {MIN_UPLOAD_ROWS} rows for statistical utility & privacy attack evaluation.",
        )
    if raw_cols < 2:
        col_name = f"'{clean_df.columns[0]}'" if raw_cols == 1 else "none"
        raise HTTPException(
            status_code=400,
            detail=f"Only {raw_cols} usable column detected ({col_name}). Please ensure your CSV has at least 2 columns separated by commas, semicolons, or tabs.",
        )

    return _upload_payload(upload_id, file.filename, clean_df, meta, [])


@app.post("/api/v1/forge", response_model=ForgeResponse)
def forge(req: ForgeRequest):
    try:
        return run_forge(req)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))


class AbortRequest(BaseModel):
    run_id: str


class PiiAction(BaseModel):
    column: str
    action: str


class PiiApplyRequest(BaseModel):
    upload_id: str
    actions: List[PiiAction]


@app.post("/api/v1/scan-pii")
def scan_pii(upload_id: str):
    try:
        frame = get_data_source(SourceSpec(type="upload", upload_id=upload_id)).load(
            SourceSpec(type="upload", upload_id=upload_id)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"upload_id": upload_id, "findings": pii_scanner.scan(frame)}


@app.post("/api/v1/apply-pii")
def apply_pii(body: PiiApplyRequest):
    spec = SourceSpec(type="upload", upload_id=body.upload_id)
    try:
        frame = get_data_source(spec).load(spec)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    updated = pii_scanner.apply(frame, [item.model_dump() for item in body.actions])
    try:
        upload_id, clean_df, meta = replace_upload(body.upload_id, updated)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    _reject_small_upload(clean_df)
    return _upload_payload(upload_id, meta.get("source_page"), clean_df, meta, [])


@app.post("/api/v1/forge/abort")
@app.post("/forge/abort")
def abort_run_endpoint(body: AbortRequest):
    """Abort an in-flight run immediately, ensuring no synthetic dataset CSV is saved."""
    success = abort_run(body.run_id)
    return {"status": "ABORTED" if success else "NOT_FOUND", "run_id": body.run_id}


@app.post("/start-run", response_model=RunStartResponse, status_code=202)
@app.post("/api/v1/start-run", response_model=RunStartResponse, status_code=202)
def start_forge_run(req: ForgeRequest):
    try:
        return start_run(req)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/status/{run_id}", response_model=RunStatusResponse)
@app.get("/api/v1/status/{run_id}", response_model=RunStatusResponse)
def run_status(run_id: str):
    status = get_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'.")
    return status


@app.get("/scorecard/{run_id}")
@app.get("/api/v1/scorecard/{run_id}")
def scorecard(run_id: str):
    status = get_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'.")
    if status.response is None:
        return {"run_id": run_id, "status": status.status, "scorecard": None}
    return {
        "run_id": run_id,
        "status": status.status,
        "scorecard": status.response.scorecard,
        "summary": status.response.scorecard_summary,
    }


@app.get("/api/v1/forge/stream")
def forge_stream(
    dataset_key: Optional[str] = None,
    upload_id: Optional[str] = None,
    n_rows: Optional[int] = None,
    max_rounds: Optional[int] = None,
    label: Optional[str] = None,
    quasi_identifiers: Optional[str] = None,
    sensitive_columns: Optional[str] = None,
    auto_heal: bool = True,
    synthesis_mode: str = "hybrid",
    dp_epsilon: float = 1.0,
    fidelity_min: Optional[float] = None,
    tstr_min: Optional[float] = None,
    query_equivalence_min: Optional[float] = None,
    reid_max: Optional[float] = None,
    membership_max: Optional[float] = None,
    attribute_inference_max: Optional[float] = None,
    linkage_max: Optional[float] = None,
    near_duplicate_max: Optional[float] = None,
    policy_name: str = "Balanced",
    policy_source: str = "preset",
    policy_explanation: str = "",
):
    """Server-Sent Events stream of the round-by-round adversarial duel.

    Works for both preset datasets (`dataset_key`) and user uploads
    (`upload_id`). For uploads, the UI may pass column-role overrides."""
    settings = get_settings()
    threshold_values = {
        "max_rounds": max_rounds if max_rounds is not None else settings.max_rounds,
        "fidelity_min": fidelity_min if fidelity_min is not None else settings.fidelity_min,
        "tstr_min": tstr_min if tstr_min is not None else settings.tstr_min,
        "query_equivalence_min": (
            query_equivalence_min
            if query_equivalence_min is not None
            else settings.query_equivalence_min
        ),
        "reid_max": reid_max if reid_max is not None else settings.reid_max,
        "membership_max": (
            membership_max if membership_max is not None else settings.membership_max
        ),
    }
    optional_privacy = {
        "attribute_inference_max": attribute_inference_max,
        "linkage_max": linkage_max,
        "near_duplicate_max": near_duplicate_max,
    }
    threshold_values.update({key: value for key, value in optional_privacy.items() if value is not None})
    try:
        thresholds = Thresholds(**threshold_values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid release policy: {exc}")
    if upload_id:
        source = SourceSpec(type="upload", upload_id=upload_id)
    elif dataset_key:
        source = SourceSpec(type="csv", dataset_key=dataset_key)
    else:
        raise HTTPException(status_code=400, detail="Provide dataset_key or upload_id.")
    if synthesis_mode not in {"hybrid", "dp"}:
        raise HTTPException(status_code=400, detail="synthesis_mode must be hybrid or dp.")
    if dp_epsilon <= 0 or dp_epsilon > 20:
        raise HTTPException(status_code=400, detail="dp_epsilon must be greater than 0 and at most 20.")

    req = ForgeRequest(
        source=source,
        n_rows=n_rows,
        thresholds=thresholds,
        label_column=label,
        quasi_identifiers=_split_csv_param(quasi_identifiers),
        sensitive_columns=_split_csv_param(sensitive_columns),
        auto_heal_enabled=auto_heal,
        synthesis_mode=synthesis_mode,
        dp_epsilon=dp_epsilon,
        policy_name=policy_name[:80],
        policy_source=policy_source[:30],
        policy_explanation=policy_explanation[:500],
    )

    def event_generator():
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                run_forge(req, progress=q.put)
            except Exception as exc:  # noqa: BLE001 - surface errors to the client
                logger.exception("Forge execution error in worker: %s", exc)
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)  # sentinel

        threading.Thread(target=worker, daemon=True).start()

        while True:
            try:
                ev = q.get(timeout=8.0)
            except queue.Empty:
                # Send SSE comment heartbeat to prevent proxy & browser idle timeouts
                yield ": keep-alive\n\n"
                continue

            if ev is None:
                break
            safe_ev = _sanitize_for_json(ev)
            yield f"data: {json.dumps(safe_ev, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _uncertified_download_detail(run_id: str) -> Optional[str]:
    """Refuse a file hand-off when this process still knows the run failed the gate."""
    status = get_run_status(run_id)
    if status is None or status.response is None or status.response.status == "SUCCESS":
        return None
    return (
        "This synthetic file is not certified, so it is not available for download."
    )


_CANDIDATE_ID = re.compile(r"^r\d+-c\d+$")


@app.post("/api/v1/forge/{run_id}/active-candidate")
def set_active_candidate(run_id: str, body: ActiveCandidateRequest):
    """Point export and BigQuery at another candidate that already passed the gate."""
    blocked = _uncertified_download_detail(run_id)
    if blocked:
        raise HTTPException(status_code=403, detail=blocked)
    if not _CANDIDATE_ID.fullmatch(body.candidate_id or ""):
        raise HTTPException(status_code=404, detail="That candidate did not pass the release gate.")
    frame = activate_passing_candidate(run_id, body.candidate_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="That candidate did not pass the release gate.")
    return {"run_id": run_id, "candidate_id": body.candidate_id, "rows": int(len(frame))}


def _require_certificate(run_id: str) -> dict:
    blocked = _uncertified_download_detail(run_id)
    if blocked:
        raise HTTPException(status_code=403, detail=blocked)
    stored = certificate.load_certificate(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="No certificate was issued for this run.")
    return stored


@app.get("/api/v1/certificate/{run_id}")
def certificate_json(run_id: str):
    return _require_certificate(run_id)


@app.get("/api/v1/certificate/{run_id}/pdf")
def certificate_pdf(run_id: str):
    _require_certificate(run_id)
    path = certificate.certificate_pdf(run_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No certificate was issued for this run.")
    return FileResponse(path, media_type="application/pdf", filename=f"certificate_{run_id}.pdf")


@app.get("/api/v1/verify/{run_id}")
def certificate_lookup(run_id: str):
    return _require_certificate(run_id)


@app.post("/api/v1/verify")
async def verify_certificate(
    file: UploadFile = File(...),
    certificate_json: Optional[str] = Form(None),
    certificate_file: Optional[UploadFile] = File(None),
):
    if certificate_file is not None:
        raw_certificate = await certificate_file.read()
    elif isinstance(certificate_json, str) and certificate_json.strip():
        raw_certificate = certificate_json.encode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Upload the certificate PDF, or provide certificate_json.")
    try:
        stored = certificate.load_certificate_document(raw_certificate)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return certificate.verify(await file.read(), stored)


@app.get("/api/v1/export/{run_id}")
def export(run_id: str, format: str = "csv", candidate_id: Optional[str] = None):
    blocked = _uncertified_download_detail(run_id)
    if blocked:
        raise HTTPException(status_code=403, detail=blocked)
    if candidate_id:
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise HTTPException(status_code=404, detail="That candidate did not pass the release gate.")
        frame = get_passing_frame(run_id, candidate_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="That candidate did not pass the release gate.")
        path = BASE_DIR / "data" / "output" / f"{run_id}_{candidate_id}.csv"
        frame.to_csv(path, index=False)
        return FileResponse(
            path,
            media_type="text/csv",
            filename=f"synthetic_{run_id}_{candidate_id}.csv",
        )
    frame = get_run_frame(run_id)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'.")
    path = BASE_DIR / "data" / "output" / f"{run_id}.csv"
    if not path.exists():
        frame.to_csv(path, index=False)
    return FileResponse(path, media_type="text/csv", filename=f"synthetic_{run_id}.csv")


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Synthetic Data Forge</h1><p>UI not found.</p>")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
