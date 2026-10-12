import asyncio
import io
import json
import unittest

import pandas as pd
from fastapi import HTTPException
from fastapi.datastructures import UploadFile
from openpyxl import Workbook
from starlette.datastructures import Headers

import certificate
import pii_scanner
from ingest import IngestError, load_tabular
from main import upload
from mcp_server import (
    synthgauntlet_get_scorecard,
    synthgauntlet_start_synthesis,
    synthgauntlet_verify_certificate,
)


def _rows(count=20):
    return pd.DataFrame({
        "age": [20 + i for i in range(count)],
        "income": [30000 + i * 100 for i in range(count)],
    })


class IngestTests(unittest.TestCase):
    def test_json_flattens_nested_fields(self):
        payload = [{"person": {"city": "Austin"}, "tags": ["a", "b"], "income": 10}]
        frame, warnings = load_tabular(json.dumps(payload).encode(), "people.json")
        self.assertIn("person_city", frame.columns)
        self.assertEqual(frame.loc[0, "tags"], "a, b")
        self.assertTrue(any("flattened" in warning for warning in warnings))

    def test_excel_uses_the_first_sheet_and_warns(self):
        book = Workbook()
        first = book.active
        first.title = "Patients"
        first.append(["age", "income"])
        first.append([40, 80])
        extra = book.create_sheet("Notes")
        extra.append(["age", "income"])
        extra.append([1, 2])
        raw = io.BytesIO()
        book.save(raw)
        frame, warnings = load_tabular(raw.getvalue(), "clinic.xlsx")
        self.assertEqual(list(frame.columns), ["age", "income"])
        self.assertEqual(int(frame.loc[0, "age"]), 40)
        self.assertTrue(any("Notes" in warning for warning in warnings))

    def test_legacy_xls_and_corrupt_files_are_rejected(self):
        with self.assertRaises(IngestError):
            load_tabular(b"not a workbook", "old.xls")
        with self.assertRaises(IngestError):
            load_tabular(b"", "empty.csv")
        with self.assertRaises(IngestError):
            load_tabular(b"\x00\x01", "broken.parquet")


class UploadFormatTests(unittest.TestCase):
    def test_json_upload_and_png_rejection(self):
        frame = _rows()
        payload = frame.to_json(orient="records").encode()
        uploaded = UploadFile(
            file=io.BytesIO(payload),
            filename="people.json",
            headers=Headers({"content-type": "application/json"}),
        )
        result = asyncio.run(upload(uploaded))
        self.assertEqual(result["rows"], 20)
        self.assertIn("age", result["columns"])

        image = UploadFile(
            file=io.BytesIO(b"fake image content"),
            filename="image.png",
            headers=Headers({"content-type": "image/png"}),
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(upload(image))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("csv", ctx.exception.detail.lower())


class PiiTests(unittest.TestCase):
    def test_scan_redacts_examples_and_apply_removes_raw_values(self):
        frame = pd.DataFrame({
            "email": ["ada@example.com"] * 20,
            "card": ["4111111111111111"] * 20,
            "age": list(range(20, 40)),
        })
        findings = {item["kind"]: item for item in pii_scanner.scan(frame)}
        self.assertIn("email", findings)
        self.assertNotIn("ada@example.com", findings["email"]["redacted_example"])
        self.assertEqual(findings["credit_card"]["kind"], "credit_card")
        masked = pii_scanner.apply(frame, [{"column": "email", "action": "mask"}, {"column": "card", "action": "drop"}])
        self.assertNotIn("card", masked.columns)
        self.assertTrue((masked["email"] == "[REDACTED]").all())


class CertificateTests(unittest.TestCase):
    def test_signature_and_fingerprint_detect_tampering(self):
        frame = _rows()
        issued = certificate.issue(
            "certtest",
            {"final_status": "CERTIFIED", "formal_dp_guarantee": "No formal DP guarantee", "policy_name": "Balanced", "rounds": [], "thresholds": {}},
            frame,
            frame,
        )
        raw = frame.to_csv(index=False).encode("utf-8")
        self.assertEqual(certificate.verify(raw, issued)["status"], "VALID")
        tampered = certificate.verify(raw.replace(b"20", b"99", 1), issued)
        self.assertEqual(tampered["status"], "TAMPERED")
        self.assertIn("fingerprint", tampered["reason"].lower())
        forged = dict(issued)
        forged["final_status"] = "CERTIFIED"
        forged["dataset_rows"] = 1
        self.assertEqual(certificate.verify(raw, forged)["status"], "TAMPERED")
        pdf = certificate.certificate_pdf("certtest").read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        from_pdf = certificate.load_certificate_document(pdf)
        self.assertEqual(certificate.verify(raw, from_pdf)["status"], "VALID")
        self.assertIn("technical evidence", issued["evidence_statement"])
        self.assertNotIn("HIPAA compliant", issued["evidence_statement"])


class SdkTests(unittest.TestCase):
    def test_mcp_tools_return_summaries_only(self):
        names = [
            synthgauntlet_start_synthesis.__name__,
            synthgauntlet_get_scorecard.__name__,
            synthgauntlet_verify_certificate.__name__,
        ]
        self.assertEqual(names, [
            "synthgauntlet_start_synthesis",
            "synthgauntlet_get_scorecard",
            "synthgauntlet_verify_certificate",
        ])
        from mcp_server import build_server
        server = build_server()
        self.assertIsNotNone(server)


if __name__ == "__main__":
    unittest.main()
