"""Data ingestion layer.

Abstracts *where* the tabular data comes from so the rest of the engine is
source-agnostic. Today: static CSVs registered in `constants.py`. Later: drop
in `BigQueryDataSource` without touching the pipeline.
"""

from __future__ import annotations

import datetime
import json
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants import DATASETS
from models import SourceSpec

# Directory where user-uploaded CSVs are persisted for the run.
UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"

# Heuristic keyword banks for auto-detecting column roles in arbitrary CSVs.
_QI_KEYWORDS = (
    "age", "gender", "sex", "zip", "zipcode", "postcode", "race", "ethnic",
    "marital", "education", "country", "nationality", "occupation", "job",
    "dob", "birth", "city", "state", "region", "department", "religion",
    "language", "pregnan", "workclass", "relationship",
)
_SENSITIVE_KEYWORDS = (
    "income", "salary", "wage", "diagnos", "disease", "condition", "disorder",
    "credit", "default", "balance", "loan", "debt", "hiv", "cancer", "mental",
    "outcome", "target", "churn", "attrition", "label", "class", "result",
    "score", "risk", "fraud", "arrest", "conviction",
)


def _looks_like_id(name: str) -> bool:
    n = name.lower()
    return n in ("id", "index", "uuid") or n.endswith("_id") or n.endswith("id")


class DatasetMeta(dict):
    """Convenience view over a constants.DATASETS entry."""

    @property
    def domain(self) -> str:
        return self.get("domain", "unknown")

    @property
    def quasi_identifiers(self) -> List[str]:
        return list(self.get("quasi_identifiers", []))

    @property
    def sensitive_columns(self) -> List[str]:
        return list(self.get("sensitive_columns", []))

    @property
    def label_column(self) -> Optional[str]:
        return self.get("label_column")


class DataSource(ABC):
    @abstractmethod
    def load(self, spec: SourceSpec) -> pd.DataFrame: ...

    @abstractmethod
    def meta(self, spec: SourceSpec) -> DatasetMeta: ...


class CsvDataSource(DataSource):
    """Reads the static demo CSVs described in constants.DATASETS."""

    def meta(self, spec: SourceSpec) -> DatasetMeta:
        key = spec.dataset_key
        if key not in DATASETS:
            raise KeyError(
                f"Unknown dataset_key '{key}'. Available: {list(DATASETS)}"
            )
        return DatasetMeta(DATASETS[key])

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        meta = self.meta(spec)
        path = meta["path"]
        if not path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {path}")
        df = pd.read_csv(path, skipinitialspace=True)
        return _clean_frame(df)


def _json_serial(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _normalize_cell(v: Any) -> Any:
    """Normalize unhashable objects (deeply nested arrays, structs, lists, dicts) into clean JSON strings."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.ndarray, list, dict, set, tuple)):
        try:
            return json.dumps(v, default=_json_serial)
        except Exception:
            return str(v)
    return v


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize unhashable/nested columns (e.g. BigQuery REPEATED or STRUCT columns like numpy.ndarray or lists)
    for col in df.columns:
        try:
            df[col].nunique(dropna=False)
        except TypeError:
            df[col] = df[col].map(_normalize_cell)
            try:
                df[col].nunique(dropna=False)
            except TypeError:
                df[col] = df[col].astype(str)

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    
    # Use dropna=False so a column containing NaNs alongside real values
    # is counted correctly and not mis-labelled as constant (nunique=1).
    try:
        nunique = df.nunique(dropna=False)
    except TypeError:
        for col in df.columns:
            try:
                df[col].nunique(dropna=False)
            except TypeError:
                df[col] = df[col].astype(str)
        nunique = df.nunique(dropna=False)

    constant_cols = nunique[nunique <= 1].index.tolist()
    if len(df.columns) - len(constant_cols) >= 2:
        return df.drop(columns=constant_cols, errors="ignore")
    return df


def infer_metadata(df: pd.DataFrame) -> dict:
    """Best-effort auto-detection of column roles for an arbitrary CSV.

    Heuristics (no raw data leaves the box):
      - label: last low-cardinality (2-20 unique) non-id column -- the natural
        prediction target; used for stratified TSTR utility.
      - quasi-identifiers: demographic-named columns + low-cardinality
        categoricals (the linkable attack surface). Capped for tractability.
      - sensitive: columns whose names imply protected/target attributes.
    """
    cols = list(df.columns)
    n = len(df)
    try:
        nunique = df.nunique()
    except TypeError:
        for col in df.columns:
            try:
                df[col].nunique()
            except TypeError:
                df[col] = df[col].astype(str)
        nunique = df.nunique()

    # --- label: prefer the last categorical-ish column ---
    label = None
    for c in reversed(cols):
        if _looks_like_id(c):
            continue
        u = int(nunique[c])
        is_floaty = pd.api.types.is_float_dtype(df[c]) and u > 20
        if 2 <= u <= 20 and not is_floaty:
            label = c
            break

    # --- sensitive: keyword match ---
    sensitive = [c for c in cols if any(k in c.lower() for k in _SENSITIVE_KEYWORDS)]

    # --- quasi-identifiers: keyword demographics + low-card categoricals ---
    qis: List[str] = []
    for c in cols:
        if c == label or _looks_like_id(c):
            continue
        name_hit = any(k in c.lower() for k in _QI_KEYWORDS)
        u = int(nunique[c])
        low_card_cat = (not pd.api.types.is_numeric_dtype(df[c])) and (2 <= u <= max(20, int(0.1 * n)))
        small_numeric = pd.api.types.is_numeric_dtype(df[c]) and u <= 30
        if name_hit or low_card_cat or small_numeric:
            qis.append(c)

    # Prefer demographic-named QIs; cap to keep the attack tractable.
    qis = sorted(set(qis), key=lambda c: (0 if any(k in c.lower() for k in _QI_KEYWORDS) else 1, cols.index(c)))
    qis = [c for c in qis if c not in sensitive][:6]

    # Fallback: if nothing detected, use the first couple of non-id columns.
    if not qis:
        qis = [c for c in cols if c != label and not _looks_like_id(c)][:3]

    return {
        "domain": "custom upload",
        "quasi_identifiers": qis,
        "sensitive_columns": sensitive,
        "label_column": label,
    }


# Registry mapping upload_id -> {"path": Path, "meta": dict}.
_UPLOAD_REGISTRY: Dict[str, dict] = {}


def register_upload(df: pd.DataFrame, filename: str = "upload.csv") -> Tuple[str, pd.DataFrame, dict]:
    """Persist a cleaned uploaded frame and infer its metadata."""
    df = _clean_frame(df)
    meta = infer_metadata(df)
    meta["source_page"] = filename

    upload_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{upload_id}.csv"
    df.to_csv(path, index=False)

    _UPLOAD_REGISTRY[upload_id] = {"path": path, "meta": meta}
    return upload_id, df, meta


class UploadCsvDataSource(DataSource):
    """Reads a user-uploaded CSV registered via `register_upload`."""

    def _entry(self, spec: SourceSpec) -> dict:
        uid = spec.upload_id
        if uid not in _UPLOAD_REGISTRY:
            raise KeyError(
                f"Unknown upload_id '{uid}'. Upload the CSV again (uploads reset on restart)."
            )
        return _UPLOAD_REGISTRY[uid]

    def meta(self, spec: SourceSpec) -> DatasetMeta:
        return DatasetMeta(self._entry(spec)["meta"])

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        path = self._entry(spec)["path"]
        if not path.exists():
            raise FileNotFoundError(f"Uploaded CSV not found at {path}")
        return _clean_frame(pd.read_csv(path, skipinitialspace=True))


class BigQueryDataSource(DataSource):
    """GCP-native BigQuery source using application default credentials."""

    _TABLE_ID = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){1,2}$")

    def _table_id(self, spec: SourceSpec) -> str:
        table_id = spec.table_id or ""
        if not self._TABLE_ID.fullmatch(table_id):
            raise ValueError("table_id must be project.dataset.table or dataset.table")
        return table_id

    def load(self, spec: SourceSpec) -> pd.DataFrame:
        from google.cloud import bigquery

        client = bigquery.Client()
        table = client.get_table(self._table_id(spec))
        return client.list_rows(table).to_dataframe()

    def meta(self, spec: SourceSpec) -> DatasetMeta:
        frame = _clean_frame(self.load(spec))
        metadata = infer_metadata(frame)
        metadata.update({"domain": "bigquery", "source_page": spec.table_id})
        return DatasetMeta(metadata)


_SOURCES: Dict[str, DataSource] = {
    "csv": CsvDataSource(),
    "upload": UploadCsvDataSource(),
    "bigquery": BigQueryDataSource(),
}


def get_data_source(spec: SourceSpec) -> DataSource:
    if spec.type not in _SOURCES:
        raise ValueError(f"Unsupported source type '{spec.type}'.")
    return _SOURCES[spec.type]


def list_datasets() -> List[dict]:
    """Lightweight catalog for the UI / API."""
    out = []
    for key, meta in DATASETS.items():
        out.append(
            {
                "key": key,
                "domain": meta.get("domain"),
                "quasi_identifiers": meta.get("quasi_identifiers", []),
                "sensitive_columns": meta.get("sensitive_columns", []),
                "label_column": meta.get("label_column"),
                "source_page": meta.get("source_page"),
                "available": meta["path"].exists(),
                "display_name": meta.get("display_name"),
                "rows": meta.get("rows"),
                "columns": meta.get("columns"),
                "description": meta.get("description"),
            }
        )
    return out
