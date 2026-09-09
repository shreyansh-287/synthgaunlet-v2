"""BigQuery client utilities for SynthGauntlet.

Provides lightweight wrappers to list datasets, list tables, and fetch
table rows as a pandas DataFrame. All functions degrade gracefully when
the google-cloud-bigquery package is not installed or credentials are
not available.

Design note — billing project vs data-location project
-------------------------------------------------------
BigQuery separates *where data lives* (the project that owns a dataset/table)
from *who pays for the API call* (the billing/execution project set on the
client).  When the user selects a public project such as
``bigquery-public-data``, we must:

  1. Use that project only in the *table reference* (data location).
  2. Use the user's own GCP project (from ``GCP_PROJECT`` in .env, or the
     project detected by Application Default Credentials) as the *client
     billing project*.

Passing ``project="bigquery-public-data"`` to ``bigquery.Client()`` sets that
as the billing project, which immediately fails with a 403 because the user
does not own that project.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("forge.bigquery")


def is_bq_available() -> bool:
    """Return True if google-cloud-bigquery is installed and credentials work."""
    try:
        from google.cloud import bigquery  # noqa: F401
        from google.auth import default as _auth_default

        _auth_default()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("BigQuery not available: %s", exc)
        return False


def _get_billing_project() -> Optional[str]:
    """Return the user's own GCP project to use as the client billing project.

    Priority:
      1. ``GCP_PROJECT`` in .env (explicit user setting).
      2. The project detected by Application Default Credentials (ADC).
      3. None — the client will raise if it cannot determine a project.
    """
    try:
        from config import get_settings

        user_project = (get_settings().gcp_project or "").strip()
        if user_project:
            return user_project
    except Exception:  # noqa: BLE001
        pass

    try:
        from google.auth import default as _auth_default

        _, detected = _auth_default()
        if detected:
            return detected
    except Exception:  # noqa: BLE001
        pass

    return None


def _get_client(project: Optional[str] = None, user_token: Optional[str] = None):
    """Return an authenticated BigQuery client using app's billing project or user OAuth token."""
    from google.cloud import bigquery

    billing_project = _get_billing_project() or (project if project and project != "bigquery-public-data" else None)
    if user_token:
        from google.oauth2.credentials import Credentials
        creds = Credentials(token=user_token)
        target_project = project if (project and project != "bigquery-public-data") else billing_project
        return bigquery.Client(project=target_project, credentials=creds)
    # Always use app billing project so queries/list calls run under app credentials
    return bigquery.Client(project=billing_project)


def list_bq_datasets(project: str, user_token: Optional[str] = None) -> List[str]:
    """Return a list of dataset IDs in the given GCP project."""
    try:
        client = _get_client(project, user_token=user_token)
        return [ds.dataset_id for ds in client.list_datasets(project=project)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_bq_datasets failed for project %s: %s", project, exc)
        err_msg = str(exc)
        if "403" in err_msg or "Permission" in err_msg or "Forbidden" in err_msg:
            raise RuntimeError(
                f"Access Denied (403) on project '{project}'. "
                "Sign in with a Google account having access to this project, or grant 'BigQuery Data Viewer' permission to the host service account."
            ) from exc
        raise RuntimeError(f"Could not list BigQuery datasets in '{project}': {exc}") from exc


def list_bq_tables(project: str, dataset: str, user_token: Optional[str] = None) -> List[str]:
    """Return a list of table IDs in the given dataset."""
    try:
        client = _get_client(project, user_token=user_token)
        dataset_ref = client.dataset(dataset, project=project)
        return [tbl.table_id for tbl in client.list_tables(dataset_ref)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_bq_tables failed for %s.%s: %s", project, dataset, exc)
        err_msg = str(exc)
        if "403" in err_msg or "Permission" in err_msg or "Forbidden" in err_msg:
            raise RuntimeError(
                f"Access Denied (403) on dataset '{project}.{dataset}'."
            ) from exc
        raise RuntimeError(f"Could not list tables in '{dataset}': {exc}") from exc


def get_bq_table_info(project: str, dataset: str, table: str, user_token: Optional[str] = None) -> dict:
    """Return metadata about a BigQuery table, including total row count."""
    try:
        client = _get_client(project, user_token=user_token)
        table_ref = client.dataset(dataset, project=project).table(table)
        tbl = client.get_table(table_ref)
        num_rows = int(tbl.num_rows) if tbl.num_rows is not None else 0
        return {
            "project": project,
            "dataset": dataset,
            "table": table,
            "total_rows": num_rows,
            "max_fetch_limit": 10000,
            "default_fetch": min(num_rows, 10000) if num_rows > 0 else 10000
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_bq_table_info failed for %s.%s.%s: %s", project, dataset, table, exc)
        return {
            "project": project,
            "dataset": dataset,
            "table": table,
            "total_rows": 0,
            "max_fetch_limit": 10000,
            "default_fetch": 10000
        }


def fetch_bq_table(project: str, dataset: str, table: str, limit: int = 5000, user_token: Optional[str] = None) -> pd.DataFrame:
    """Download up to ``limit`` rows from a BigQuery table as a DataFrame.

    Uses ``client.list_rows()`` (TableData API) which does not create a
    BigQuery query job and therefore works on publicly accessible tables
    without requiring ``bigquery.jobs.create`` permissions.
    """
    logger.info(
        "fetch_bq_table: data_project=%s dataset=%s table=%s limit=%d",
        project, dataset, table, limit,
    )

    client = _get_client(project, user_token=user_token)
    table_ref = client.dataset(dataset, project=project).table(table)

    try:
        logger.info("Attempting list_rows for %s.%s.%s", project, dataset, table)
        df = client.list_rows(table_ref, max_results=int(limit)).to_dataframe()
        logger.info(
            "list_rows succeeded: fetched %d rows from %s.%s.%s",
            len(df), project, dataset, table,
        )
        return df

    except Exception as list_exc:  # noqa: BLE001
        logger.warning(
            "list_rows failed for %s.%s.%s — original error: %s",
            project, dataset, table, list_exc,
        )
        raise RuntimeError(
            f"Could not import {project}.{dataset}.{table} via the TableData API (list_rows). "
            f"Original error: {list_exc}. "
            f"If this object is a SQL VIEW, set GCP_PROJECT in your .env to a project where you have "
            f"bigquery.jobs.create permission, then retry."
        ) from list_exc


def export_synthetic_to_bq(
    df: pd.DataFrame,
    project: str,
    dataset: str,
    table: str,
) -> dict:
    """Export a synthetic pandas DataFrame into a BigQuery table (WRITE_TRUNCATE).

    Creates the destination dataset if it does not exist, and overwrites/creates the table.
    """
    if not is_bq_available():
        raise RuntimeError("BigQuery client is not available or GCP credentials are not logged in.")

    import re
    if not project or not re.match(r"^[A-Za-z0-9_-]+$", project):
        raise ValueError(f"Invalid GCP project ID: '{project}'")
    if not dataset or not re.match(r"^[A-Za-z0-9_]+$", dataset):
        raise ValueError(f"Invalid dataset ID: '{dataset}'")
    if not table or not re.match(r"^[A-Za-z0-9_]+$", table):
        raise ValueError(f"Invalid table ID: '{table}'")

    if df is None or df.empty:
        raise ValueError("Cannot export an empty dataset to BigQuery.")

    # Sanitize DataFrame columns for BigQuery compliance
    export_df = df.copy()
    cleaned_cols = []
    for col in export_df.columns:
        c_str = str(col).strip()
        c_clean = re.sub(r"[^\w]", "_", c_str)
        if not c_clean or c_clean[0].isdigit():
            c_clean = f"col_{c_clean}"
        cleaned_cols.append(c_clean)
    export_df.columns = cleaned_cols

    from google.cloud import bigquery
    client = _get_client(project)

    # 1. Ensure target dataset exists (default US location)
    dataset_ref = bigquery.DatasetReference(project, dataset)
    try:
        dataset_obj = bigquery.Dataset(dataset_ref)
        dataset_obj.location = "US"
        client.create_dataset(dataset_obj, exists_ok=True)
    except Exception as ds_exc:
        logger.warning("create_dataset warning for %s.%s: %s", project, dataset, ds_exc)

    # 2. Configure load job to write DataFrame into BigQuery table
    table_ref = dataset_ref.table(table)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    try:
        load_job = client.load_table_from_dataframe(export_df, table_ref, job_config=job_config)
        load_job.result(timeout=60)

        full_table_id = f"{project}.{dataset}.{table}"
        logger.info("Successfully exported %d synthetic rows to BigQuery: %s", len(export_df), full_table_id)
        return {
            "success": True,
            "project": project,
            "dataset": dataset,
            "table": table,
            "table_id": full_table_id,
            "rows": int(len(export_df)),
            "columns": list(export_df.columns),
        }
    except Exception as exc:
        logger.exception("export_synthetic_to_bq failed for %s.%s.%s: %s", project, dataset, table, exc)
        err_msg = str(exc)
        if "403" in err_msg or "Permission" in err_msg:
            raise RuntimeError(
                f"Permission Denied (403) while writing to '{project}.{dataset}.{table}'. "
                "Ensure your GCP Service Account has 'BigQuery Data Editor' and 'BigQuery Job User' roles on this project."
            ) from exc
        raise RuntimeError(f"Failed to export synthetic data to BigQuery: {exc}") from exc


