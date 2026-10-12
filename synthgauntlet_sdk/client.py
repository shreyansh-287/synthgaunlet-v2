"""HTTP client for upload, synthesis, scorecard, download, and certificate checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

from synthgauntlet_sdk.models import Scorecard, Verification


class SynthGauntletClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def upload(self, path: str | Path, sheet_name: Optional[str] = None) -> dict:
        file_path = Path(path)
        fields = {"sheet_name": sheet_name} if sheet_name else {}
        return self._multipart("/api/v1/upload", file_path, fields)

    def run_forge(self, source: dict, **options) -> dict:
        body = {"source": source, **options}
        return self._json("POST", "/api/v1/forge", body)

    def get_scorecard(self, run_id: str) -> Scorecard:
        payload = self._json("GET", f"/api/v1/status/{run_id}")
        response = payload.get("response") or {}
        return Scorecard(
            run_id=payload.get("run_id", run_id),
            status=payload.get("status", ""),
            final_status=response.get("final_status", ""),
            summary=response.get("scorecard_summary", ""),
            metrics=_metric_summary(response),
        )

    def download(self, run_id: str, destination: str | Path) -> Path:
        target = Path(destination)
        request = Request(f"{self.base_url}/api/v1/export/{run_id}?format=csv")
        with urlopen(request) as response:
            target.write_bytes(response.read())
        return target

    def verify(self, csv_path: str | Path, certificate_path: str | Path) -> Verification:
        csv_file = Path(csv_path)
        certificate = json.loads(Path(certificate_path).read_text(encoding="utf-8"))
        payload = self._multipart(
            "/api/v1/verify",
            csv_file,
            {"certificate_json": json.dumps(certificate)},
        )
        return Verification(
            status=payload.get("status", "TAMPERED"),
            reason=payload.get("reason", ""),
            run_id=payload.get("run_id", ""),
        )

    def _json(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def _multipart(self, path: str, file_path: Path, fields: dict) -> dict:
        boundary = "----synthgauntlet"
        chunks = []
        for key, value in fields.items():
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
            )
        file_bytes = file_path.read_bytes()
        chunks.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{file_path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + file_bytes
            + b"\r\n"
        )
        chunks.append(f"--{boundary}--\r\n".encode())
        request = Request(
            f"{self.base_url}{path}",
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))


def _metric_summary(payload: dict) -> dict:
    rounds = payload.get("rounds") or []
    final = rounds[-1] if rounds else {}
    utility = final.get("utility") or {}
    attack = final.get("attack") or {}
    return {
        "fidelity_score": utility.get("fidelity_score"),
        "utility_ratio": utility.get("utility_ratio"),
        "re_id_rate": attack.get("re_id_rate"),
        "membership_accuracy": attack.get("membership_accuracy"),
    }
