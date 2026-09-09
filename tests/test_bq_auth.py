import unittest
from unittest.mock import MagicMock, patch

from main import auth_config, bq_datasets, bq_tables, bq_table_info, bq_import, BQImportRequest, _extract_token
import bigquery_client as bq


class TestBigQueryAuth(unittest.TestCase):
    def test_extract_token_variations(self):
        # Valid bearer token
        self.assertEqual(_extract_token("Bearer ya29.test_token_123"), "ya29.test_token_123")
        self.assertEqual(_extract_token("Bearer  token_with_spaces  "), "token_with_spaces")

        # Invalid or missing tokens
        self.assertIsNone(_extract_token(None))
        self.assertIsNone(_extract_token(""))
        self.assertIsNone(_extract_token("Basic dXNlcjpwYXNz"))
        self.assertIsNone(_extract_token("Bearer "))
        self.assertIsNone(_extract_token("Bearer null"))
        self.assertIsNone(_extract_token("Bearer undefined"))
        self.assertIsNone(_extract_token("Bearer none"))

    def test_auth_config_endpoint(self):
        res = auth_config()
        self.assertIsInstance(res, dict)
        self.assertIn("google_client_id", res)

    @patch("bigquery_client.is_bq_available", return_value=True)
    @patch("bigquery_client.list_bq_datasets")
    def test_bq_datasets_with_bearer_token(self, mock_list_ds, mock_avail):
        mock_list_ds.return_value = ["dataset_a", "dataset_b"]

        res = bq_datasets(
            project="private-client-project",
            authorization="Bearer ya29.custom_user_oauth_token"
        )
        self.assertEqual(res["datasets"], ["dataset_a", "dataset_b"])
        mock_list_ds.assert_called_once_with("private-client-project", user_token="ya29.custom_user_oauth_token")

    @patch("bigquery_client.is_bq_available", return_value=True)
    @patch("bigquery_client.list_bq_tables")
    def test_bq_tables_with_bearer_token(self, mock_list_tbls, mock_avail):
        mock_list_tbls.return_value = ["table_1", "table_2"]

        res = bq_tables(
            project="private-client-project",
            dataset="dataset_a",
            authorization="Bearer ya29.custom_user_oauth_token"
        )
        self.assertEqual(res["tables"], ["table_1", "table_2"])
        mock_list_tbls.assert_called_once_with("private-client-project", "dataset_a", user_token="ya29.custom_user_oauth_token")

    @patch("bigquery_client.is_bq_available", return_value=True)
    @patch("bigquery_client.get_bq_table_info")
    def test_bq_table_info_with_bearer_token(self, mock_info, mock_avail):
        mock_info.return_value = {
            "project": "private-client-project",
            "dataset": "dataset_a",
            "table": "table_1",
            "total_rows": 5420,
            "max_fetch_limit": 10000,
            "default_fetch": 5420
        }

        res = bq_table_info(
            project="private-client-project",
            dataset="dataset_a",
            table="table_1",
            authorization="Bearer ya29.custom_user_oauth_token"
        )
        self.assertEqual(res["total_rows"], 5420)
        mock_info.assert_called_once_with("private-client-project", "dataset_a", "table_1", user_token="ya29.custom_user_oauth_token")

    @patch("bigquery_client.is_bq_available", return_value=True)
    @patch("bigquery_client.get_bq_table_info")
    @patch("bigquery_client.fetch_bq_table")
    def test_bq_import_with_user_token(self, mock_fetch, mock_info, mock_avail):
        import pandas as pd
        mock_info.return_value = {"total_rows": 500}
        mock_fetch.return_value = pd.DataFrame({"colA": range(60), "colB": range(60)})

        req = BQImportRequest(
            project="private-client-project",
            dataset="my_dataset",
            table="my_table",
            limit=1000
        )
        res = bq_import(
            body=req,
            authorization="Bearer ya29.custom_user_oauth_token"
        )
        self.assertEqual(res["rows"], 60)
        mock_fetch.assert_called_once_with("private-client-project", "my_dataset", "my_table", limit=1000, user_token="ya29.custom_user_oauth_token")


if __name__ == "__main__":
    unittest.main()
