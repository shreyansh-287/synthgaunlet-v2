"""BigQuery-backed and local file metadata history for the sidebar."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config import get_settings

logger = logging.getLogger("forge.history")

_DEFAULT_HISTORY = [
    {"id": "h-1", "name": "Health Diabetes", "details": "768 rows · Demo", "type": "sample", "key": "health_diabetes", "status": "SUCCESS"},
    {"id": "h-2", "name": "Adult Income", "details": "32,561 rows · Demo", "type": "sample", "key": "adult_income", "status": "SUCCESS"},
    {"id": "h-3", "name": "Bank Customers", "details": "1,470 rows · Demo", "type": "sample", "key": "bank_customers", "status": "READY"},
]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")

_STATUS_MAP_FILE = Path(__file__).parent / "data" / "history_status_map.json"
_LOCAL_ITEMS_FILE = Path(__file__).parent / "data" / "history_items.json"
_DELETED_IDS_FILE = Path(__file__).parent / "data" / "history_deleted_ids.json"
_CLEARED_AT_FILE = Path(__file__).parent / "data" / "history_cleared_at.json"


class HistoryItem(BaseModel):
    id: Optional[str] = None
    sessionId: Optional[str] = None
    name: str = Field(min_length=1, max_length=200)
    details: str = Field(min_length=1, max_length=500)
    type: str = Field(min_length=1, max_length=32)
    key: Optional[str] = Field(default=None, max_length=500)
    status: Optional[str] = Field(default="READY", max_length=64)


class HistoryStoreUnavailable(RuntimeError):
    pass


def _safe_item(item: HistoryItem | Dict[str, Any]) -> Dict[str, Any]:
    value = item if isinstance(item, HistoryItem) else HistoryItem.model_validate(item)
    item_id = value.id or ("h-" + hashlib.sha1(f"{value.type}:{value.key or value.name}".encode()).hexdigest()[:12])
    return {
        "id": item_id,
        "sessionId": value.sessionId,
        "name": value.name.strip(),
        "details": value.details.strip(),
        "type": value.type.strip(),
        "key": value.key.strip() if value.key else None,
        "status": (value.status or "READY").strip(),
    }


def _load_status_map() -> Dict[str, str]:
    if _STATUS_MAP_FILE.exists():
        try:
            return json.loads(_STATUS_MAP_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_status_map(m: Dict[str, str]) -> None:
    try:
        _STATUS_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATUS_MAP_FILE.write_text(json.dumps(m, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save history status map: %s", exc)


def _load_deleted_ids() -> set:
    if _DELETED_IDS_FILE.exists():
        try:
            data = json.loads(_DELETED_IDS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(data)
        except Exception:
            pass
    return set()


def _save_deleted_ids(deleted: set) -> None:
    try:
        _DELETED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DELETED_IDS_FILE.write_text(json.dumps(list(deleted), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save deleted history ids: %s", exc)


def _load_cleared_at() -> Optional[datetime]:
    if _CLEARED_AT_FILE.exists():
        try:
            raw = json.loads(_CLEARED_AT_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, str):
                return datetime.fromisoformat(raw)
        except Exception:
            pass
    return None


def _save_cleared_at(ts: Optional[datetime]) -> None:
    try:
        _CLEARED_AT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if ts is not None:
            _CLEARED_AT_FILE.write_text(json.dumps(ts.isoformat()), encoding="utf-8")
        elif _CLEARED_AT_FILE.exists():
            try:
                os.remove(str(_CLEARED_AT_FILE))
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Could not save cleared_at: %s", exc)


def _load_local_items() -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (items, is_initialized)."""
    if _LOCAL_ITEMS_FILE.exists():
        try:
            raw = json.loads(_LOCAL_ITEMS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [_safe_item(it) for it in raw], True
        except Exception:
            pass
    return [_safe_item(dict(item)) for item in _DEFAULT_HISTORY], False


def _save_local_items(items: List[Dict[str, Any]]) -> None:
    try:
        _LOCAL_ITEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LOCAL_ITEMS_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save local history items: %s", exc)


def _client_and_table():
    from google.cloud import bigquery
    from google.auth import default as auth_default

    settings = get_settings()
    project = (settings.gcp_project or "").strip()
    if not project:
        _, project = auth_default()
    if not project:
        raise HistoryStoreUnavailable("Set GCP_PROJECT or configure Application Default Credentials.")
    dataset = settings.bq_history_dataset.strip()
    table = settings.bq_history_table.strip()
    if not _IDENTIFIER.fullmatch(dataset) or not _IDENTIFIER.fullmatch(table):
        raise HistoryStoreUnavailable("BQ_HISTORY_DATASET and BQ_HISTORY_TABLE must be simple identifiers.")
    client = bigquery.Client(project=project)
    return client, bigquery, f"{project}.{dataset}.{table}", dataset


def _ensure_table(client, bigquery, table_id: str, dataset_id: str) -> None:
    from google.api_core.exceptions import NotFound

    try:
        client.get_table(table_id)
        return
    except NotFound:
        pass

    settings = get_settings()
    dataset_ref = bigquery.Dataset(f"{table_id.split('.')[0]}.{dataset_id}")
    dataset_ref.location = settings.bq_history_location
    client.create_dataset(dataset_ref, exists_ok=True)
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("details", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("item_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("item_key", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    client.create_table(bigquery.Table(table_id, schema=schema))


def _row_to_item(row: Any) -> Tuple[Dict[str, Any], Optional[datetime]]:
    values = dict(row.items())
    created_at = values.get("created_at")
    dt = None
    if isinstance(created_at, datetime):
        dt = created_at
    elif isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at)
        except Exception:
            pass
    return {
        "id": values.get("id"),
        "sessionId": values.get("sessionId") or values.get("session_id"),
        "name": values.get("name", ""),
        "details": values.get("details", ""),
        "type": values.get("item_type", ""),
        "key": values.get("item_key"),
        "status": values.get("status", "READY"),
    }, dt


def list_history() -> Tuple[List[Dict[str, Any]], bool]:
    status_map = _load_status_map()
    deleted_ids = _load_deleted_ids()
    cleared_at = _load_cleared_at()
    local_items, is_initialized = _load_local_items()

    # Filter out any local items that were marked deleted
    local_items = [it for it in local_items if it.get("id") not in deleted_ids]

    try:
        client, bigquery, table_id, dataset_id = _client_and_table()
        _ensure_table(client, bigquery, table_id, dataset_id)
        settings = get_settings()
        limit = max(1, min(settings.history_limit, 100))
        rows = list(client.query(
            f"SELECT id, name, details, item_type, item_key, created_at FROM `{table_id}` "
            f"ORDER BY created_at DESC LIMIT {limit}"
        ).result())
        
        # If BigQuery table is empty:
        if not rows:
            if is_initialized:
                for it in local_items:
                    if it.get("id") in status_map:
                        it["status"] = status_map[it["id"]]
                return local_items, True

            # First run: seed default demo history into BigQuery and local store
            items = [_safe_item(item) for item in _DEFAULT_HISTORY]
            seed_time = datetime.now(timezone.utc)
            client.insert_rows_json(table_id, [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "details": item["details"],
                    "item_type": item["type"],
                    "item_key": item["key"],
                    "created_at": (seed_time - timedelta(seconds=index)).isoformat(),
                }
                for index, item in enumerate(items)
            ])
            for it in items:
                if it["id"] in status_map:
                    it["status"] = status_map[it["id"]]
            _save_local_items(items)
            return items, True
            
        # Merge local items and BigQuery rows:
        # Keep local items first (preserves recent additions that might have BQ streaming lag)
        merged_items = []
        seen = set()
        for it in local_items:
            it_id = it.get("id")
            if it_id and it_id not in seen:
                seen.add(it_id)
                if it_id in status_map:
                    it["status"] = status_map[it_id]
                merged_items.append(it)

        for row in rows:
            item, row_dt = _row_to_item(row)
            item_id = item.get("id")
            if not item_id or item_id in seen or item_id in deleted_ids:
                continue
            if cleared_at and row_dt:
                row_dt_utc = row_dt if row_dt.tzinfo else row_dt.replace(tzinfo=timezone.utc)
                cleared_at_utc = cleared_at if cleared_at.tzinfo else cleared_at.replace(tzinfo=timezone.utc)
                if row_dt_utc <= cleared_at_utc:
                    continue

            seen.add(item_id)
            if item_id in status_map:
                item["status"] = status_map[item_id]
            merged_items.append(item)

        _save_local_items(merged_items)
        return merged_items, True
        
    except Exception as exc:  # noqa: BLE001
        logger.warning("BigQuery history unavailable; using local file store: %s", exc)
        for it in local_items:
            if it.get("id") in status_map:
                it["status"] = status_map[it.get("id")]
        return local_items, False


def save_history(item: HistoryItem | Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    normalized = _safe_item(item)

    deleted_ids = _load_deleted_ids()
    if normalized["id"] in deleted_ids:
        deleted_ids.remove(normalized["id"])
        _save_deleted_ids(deleted_ids)

    status_map = _load_status_map()
    status_map[normalized["id"]] = normalized["status"]
    _save_status_map(status_map)

    local_items, _ = _load_local_items()
    local_items = [x for x in local_items if x.get("id") not in deleted_ids]
    existing_idx = next((i for i, x in enumerate(local_items) if x.get("id") == normalized["id"]), None)
    if existing_idx is not None:
        local_items[existing_idx] = normalized
    else:
        local_items.insert(0, normalized)
    _save_local_items(local_items)

    try:
        client, bigquery, table_id, dataset_id = _client_and_table()
        _ensure_table(client, bigquery, table_id, dataset_id)
        errors = client.insert_rows_json(table_id, [{
            "id": normalized["id"],
            "name": normalized["name"],
            "details": normalized["details"],
            "item_type": normalized["type"],
            "item_key": normalized["key"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }])
        if errors:
            raise HistoryStoreUnavailable(f"Could not persist history metadata: {errors}")
        return normalized, True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist history metadata to BigQuery: %s", exc)
        return normalized, False


def default_history() -> List[Dict[str, Any]]:
    status_map = _load_status_map()
    items = [dict(item) for item in _DEFAULT_HISTORY]
    for it in items:
        if it.get("id") in status_map:
            it["status"] = status_map[it["id"]]
    return items


def clear_history() -> bool:
    """Clear all history items permanently from local status map, local items, and BigQuery table."""
    now = datetime.now(timezone.utc)
    _save_cleared_at(now)
    _save_deleted_ids(set())
    _save_status_map({})
    _save_local_items([])

    try:
        client, bigquery, table_id, _ = _client_and_table()
        client.query(f"TRUNCATE TABLE `{table_id}`").result()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not truncate BigQuery history table: %s", exc)
        return True


def delete_history_item(item_id: str) -> bool:
    """Delete a single history item by ID."""
    status_map = _load_status_map()
    if item_id in status_map:
        del status_map[item_id]
        _save_status_map(status_map)

    deleted_ids = _load_deleted_ids()
    deleted_ids.add(item_id)
    _save_deleted_ids(deleted_ids)

    local_items, _ = _load_local_items()
    filtered = [x for x in local_items if x.get("id") != item_id]
    _save_local_items(filtered)

    try:
        client, bigquery, table_id, _ = _client_and_table()
        client.query(f"DELETE FROM `{table_id}` WHERE id = '{item_id}'").result()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete item %s from BigQuery: %s", item_id, exc)
        return True
