import asyncio
import io
import unittest
from fastapi import HTTPException
from fastapi.datastructures import UploadFile
from starlette.datastructures import Headers
import pandas as pd
from main import upload


class TestUploadAPI(unittest.TestCase):
    def setUp(self):
        # Create a valid CSV with >50 rows and >=2 columns
        df = pd.DataFrame({
            "age": [20 + (i % 50) for i in range(60)],
            "income": [30000 + i * 1000 for i in range(60)],
            "target": [i % 2 for i in range(60)]
        })
        self.valid_csv_bytes = df.to_csv(index=False).encode("utf-8")

    def test_upload_valid_csv_standard(self):
        upload_file = UploadFile(
            file=io.BytesIO(self.valid_csv_bytes),
            filename="my_data.csv",
            headers=Headers({"content-type": "text/csv"})
        )
        res = asyncio.run(upload(upload_file))
        self.assertIn("upload_id", res)
        self.assertEqual(res["rows"], 60)
        self.assertIn("age", res["columns"])

    def test_upload_mobile_mime_types(self):
        for ct in ["application/octet-stream", "text/plain", "application/vnd.ms-excel", "application/csv", "text/x-csv", "text/comma-separated-values"]:
            upload_file = UploadFile(
                file=io.BytesIO(self.valid_csv_bytes),
                filename="mobile_dataset.csv",
                headers=Headers({"content-type": ct})
            )
            res = asyncio.run(upload(upload_file))
            self.assertIn("upload_id", res)
            self.assertEqual(res["rows"], 60)

    def test_upload_rejects_non_csv(self):
        upload_file = UploadFile(
            file=io.BytesIO(b"fake image content"),
            filename="image.png",
            headers=Headers({"content-type": "image/png"})
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(upload(upload_file))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
