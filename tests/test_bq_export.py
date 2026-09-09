import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi import HTTPException

import bigquery_client as bq
from main import bq_export, BQExportRequest
from pipeline import _RUN_CACHE


class TestBigQueryExport(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "User Age": [25, 30, 45],
            "Annual Income ($)": [50000, 75000, 120000],
            "123_tag": ["A", "B", "C"]
        })

    def test_export_invalid_inputs(self):
        with patch("bigquery_client.is_bq_available", return_value=True):
            # Empty dataframe
            with self.assertRaises(ValueError):
                bq.export_synthetic_to_bq(pd.DataFrame(), "test-proj", "synth_ds", "synth_tbl")

            # Invalid project
            with self.assertRaises(ValueError):
                bq.export_synthetic_to_bq(self.df, "bad/project!", "synth_ds", "synth_tbl")

            # Invalid dataset
            with self.assertRaises(ValueError):
                bq.export_synthetic_to_bq(self.df, "test-proj", "bad-dataset-with-dash", "synth_tbl")

            # Invalid table
            with self.assertRaises(ValueError):
                bq.export_synthetic_to_bq(self.df, "test-proj", "synth_ds", "bad table name")

    @patch("bigquery_client.is_bq_available", return_value=True)
    @patch("bigquery_client._get_client")
    def test_export_synthetic_to_bq_success(self, mock_get_client, mock_avail):
        mock_client = MagicMock()
        mock_load_job = MagicMock()
        mock_load_job.result.return_value = None
        mock_client.load_table_from_dataframe.return_value = mock_load_job
        mock_get_client.return_value = mock_client

        res = bq.export_synthetic_to_bq(
            df=self.df,
            project="my-gcp-project",
            dataset="synth_dataset",
            table="synthetic_tbl",
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["table_id"], "my-gcp-project.synth_dataset.synthetic_tbl")
        self.assertEqual(res["rows"], 3)
        # Check column sanitization
        expected_cols = ["User_Age", "Annual_Income____", "col_123_tag"]
        self.assertEqual(res["columns"], expected_cols)
        mock_client.load_table_from_dataframe.assert_called_once()


class TestBigQueryExportAPI(unittest.TestCase):
    def setUp(self):
        self.run_id = "test_run_bq_export_123"
        _RUN_CACHE[self.run_id] = pd.DataFrame({"age": [20, 30], "score": [0.8, 0.9]})

    def tearDown(self):
        if self.run_id in _RUN_CACHE:
            del _RUN_CACHE[self.run_id]

    def test_bq_export_missing_run(self):
        req = BQExportRequest(
            run_id="non_existent_run_xyz",
            project="test-proj",
            dataset="test_ds",
            table="test_tbl"
        )
        with self.assertRaises(HTTPException) as ctx:
            bq_export(req)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_bq_export_missing_fields(self):
        req = BQExportRequest(
            run_id=self.run_id,
            project="",
            dataset="",
            table=""
        )
        with self.assertRaises(HTTPException) as ctx:
            bq_export(req)
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("bigquery_client.is_bq_available", return_value=True)
    @patch("bigquery_client.export_synthetic_to_bq")
    def test_bq_export_endpoint_success(self, mock_export, mock_avail):
        mock_export.return_value = {
            "success": True,
            "project": "my-project",
            "dataset": "synthgauntlet_exports",
            "table": "synth_test_tbl",
            "table_id": "my-project.synthgauntlet_exports.synth_test_tbl",
            "rows": 2,
            "columns": ["age", "score"]
        }

        req = BQExportRequest(
            run_id=self.run_id,
            project="my-project",
            dataset="synthgauntlet_exports",
            table="synth_test_tbl"
        )
        data = bq_export(req)

        self.assertTrue(data["success"])
        self.assertEqual(data["table_id"], "my-project.synthgauntlet_exports.synth_test_tbl")
        self.assertEqual(data["rows"], 2)


if __name__ == "__main__":
    unittest.main()
