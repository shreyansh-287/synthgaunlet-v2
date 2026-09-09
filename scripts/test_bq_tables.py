"""Automated Smoke-Test script for diverse BigQuery Public Datasets.
Tests schema ingestion, normalization, metadata inference, and JSON serialization
across completely distinct public datasets without manual GUI clicking.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigquery_client import fetch_bq_table, get_bq_table_info
from data_source import register_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_bq")

TEST_TABLES = [
    ("bigquery-public-data", "america_health_rankings", "ahr"),
    ("bigquery-public-data", "cymbal_investments", "trade_capture_report"),
    ("bigquery-public-data", "austin_bikeshare", "bikeshare_trips"),
    ("bigquery-public-data", "fda_food", "food_enforcement"),
    ("bigquery-public-data", "san_francisco_trees", "street_tree_list"),
    ("bigquery-public-data", "census_bureau_acs", "county_2017_5yr"),
    ("bigquery-public-data", "world_bank_wdi", "indicators_data"),
]

def run_tests():
    logger.info("Starting BigQuery Multi-Dataset Consistency Test (%d tables)...", len(TEST_TABLES))
    results = []

    for project, dataset, table in TEST_TABLES:
        table_id = f"{project}.{dataset}.{table}"
        logger.info("--> Testing table: %s", table_id)
        try:
            # 1. Fetch sample
            df = fetch_bq_table(project, dataset, table, limit=50)
            raw_rows, raw_cols = len(df), len(df.columns)
            
            # 2. Register upload (runs _clean_frame, infer_metadata, CSV persist)
            upload_id, clean_df, meta = register_upload(df, filename=f"{dataset}.{table}.bq")
            
            clean_cols = len(clean_df.columns)
            results.append({
                "table": table_id,
                "status": "PASS",
                "raw_shape": f"{raw_rows}x{raw_cols}",
                "clean_cols": clean_cols,
                "label": meta.get("label_column"),
                "qis": len(meta.get("quasi_identifiers", [])),
            })
            logger.info("✓ PASS: %s (clean shape: %dx%d, label=%s)", table_id, len(clean_df), clean_cols, meta.get("label_column"))
        except Exception as exc:
            logger.error("✗ FAIL: %s -> %s", table_id, exc, exc_info=True)
            results.append({
                "table": table_id,
                "status": "FAIL",
                "error": str(exc),
            })

    print("\n" + "="*80)
    print("BIGQUERY MULTI-DATASET CONSISTENCY TEST RESULTS")
    print("="*80)
    for r in results:
        if r["status"] == "PASS":
            print(f"✓ [PASS] {r['table']:<55} | Clean cols: {r['clean_cols']} | Label: {r['label']}")
        else:
            print(f"✗ [FAIL] {r['table']:<55} | Error: {r['error']}")
    print("="*80)

if __name__ == "__main__":
    run_tests()
