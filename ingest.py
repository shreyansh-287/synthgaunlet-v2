"""Load CSV, TSV, Excel, JSON, JSONL, and Parquet uploads into one DataFrame."""

from __future__ import annotations

import io
import json
from typing import Optional

import pandas as pd

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_ROWS = 50_000
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}


class IngestError(ValueError):
    """A file is empty, corrupt, or in an unsupported format."""


def load_tabular(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    if not file_bytes:
        raise IngestError("The file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise IngestError("File too large (max 15 MB).")

    name = (filename or "").lower().strip()
    suffix = _suffix(name)
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            "Unsupported file. Upload a .csv, .tsv, .xlsx, .json, .jsonl, or .parquet file."
        )

    warnings: list[str] = []
    if suffix in {".csv", ".tsv", ".txt"}:
        frame = _read_delimited(file_bytes, "\t" if suffix == ".tsv" else None)
    elif suffix in {".xlsx", ".xls"}:
        frame, sheet_warnings = _read_excel(file_bytes, suffix, sheet_name)
        warnings.extend(sheet_warnings)
    elif suffix == ".parquet":
        frame = _read_parquet(file_bytes)
    else:
        frame, json_warnings = _read_json(file_bytes, lines=suffix == ".jsonl")
        warnings.extend(json_warnings)

    frame = _flatten_lists(frame, warnings)
    if frame.empty or frame.shape[1] == 0:
        raise IngestError("The file has no usable rows or columns.")
    if len(frame) > MAX_UPLOAD_ROWS:
        frame = frame.sample(n=MAX_UPLOAD_ROWS, random_state=42).reset_index(drop=True)
        warnings.append(f"Sampled the first deterministic {MAX_UPLOAD_ROWS:,} rows.")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame, warnings


def _suffix(filename: str) -> str:
    if filename.endswith(".jsonl"):
        return ".jsonl"
    dot = filename.rfind(".")
    return filename[dot:] if dot >= 0 else ""


def _decode(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(encoding).replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace").replace("\x00", "")


def _read_delimited(file_bytes: bytes, separator: Optional[str]) -> pd.DataFrame:
    text = _decode(file_bytes)
    try:
        if separator is None:
            return pd.read_csv(io.StringIO(text), sep=None, engine="python", skipinitialspace=True)
        return pd.read_csv(io.StringIO(text), sep=separator, skipinitialspace=True)
    except Exception as exc:
        raise IngestError(f"Could not parse the delimited file: {exc}") from exc


def _read_excel(file_bytes: bytes, suffix: str, sheet_name: Optional[str]) -> tuple[pd.DataFrame, list[str]]:
    if suffix == ".xls":
        raise IngestError("Legacy .xls workbooks are not supported. Save the file as .xlsx and upload it again.")
    try:
        book = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as exc:
        raise IngestError(f"Could not read the Excel workbook: {exc}") from exc
    sheets = [str(sheet) for sheet in book.sheet_names]
    if not sheets:
        raise IngestError("The workbook has no sheets.")
    chosen = sheet_name or sheets[0]
    if chosen not in sheets:
        raise IngestError(f"Sheet '{chosen}' was not found. Available sheets: {', '.join(sheets)}.")
    warnings = []
    if sheet_name is None and len(sheets) > 1:
        warnings.append(f"Using sheet '{chosen}'. Other sheets: {', '.join(sheets[1:])}.")
    return pd.read_excel(book, sheet_name=chosen), warnings


def _read_parquet(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(file_bytes), engine="pyarrow")
    except Exception as exc:
        raise IngestError(f"Could not read the Parquet file: {exc}") from exc


def _read_json(file_bytes: bytes, lines: bool) -> tuple[pd.DataFrame, list[str]]:
    text = _decode(file_bytes)
    try:
        if lines:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError as exc:
        raise IngestError(f"Could not parse JSON: {exc}") from exc
    if not records:
        raise IngestError("The JSON file has no records.")
    warnings = []
    if any(isinstance(record, dict) and any(isinstance(value, (dict, list)) for value in record.values()) for record in records):
        warnings.append("Nested JSON fields were flattened with '_' between key names.")
    return pd.json_normalize(records, sep="_"), warnings


def _flatten_lists(frame: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    changed = False
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, list)).any():
            frame[column] = frame[column].map(
                lambda value: ", ".join(str(item) for item in value) if isinstance(value, list) else value
            )
            changed = True
    if changed:
        warnings.append("Nested lists were converted to comma-separated text.")
    return frame
