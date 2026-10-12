"""FastMCP tools. Every tool returns a metric summary or status, never raw rows."""

from __future__ import annotations

import json
from pathlib import Path

from synthgauntlet_sdk.client import SynthGauntletClient


def synthgauntlet_start_synthesis(csv_path: str, base_url: str = "http://127.0.0.1:8080") -> dict:
    client = SynthGauntletClient(base_url)
    uploaded = client.upload(csv_path)
    response = client.run_forge({"type": "upload", "upload_id": uploaded["upload_id"]})
    return {
        "run_id": response.get("run_id", ""),
        "status": response.get("status", ""),
        "final_status": response.get("final_status", ""),
        "summary": response.get("scorecard_summary", ""),
    }


def synthgauntlet_get_scorecard(run_id: str, base_url: str = "http://127.0.0.1:8080") -> dict:
    scorecard = SynthGauntletClient(base_url).get_scorecard(run_id)
    return {
        "run_id": scorecard.run_id,
        "status": scorecard.status,
        "final_status": scorecard.final_status,
        "summary": scorecard.summary,
        "metrics": scorecard.metrics,
    }


def synthgauntlet_verify_certificate(csv_path: str, certificate_path: str, base_url: str = "http://127.0.0.1:8080") -> dict:
    result = SynthGauntletClient(base_url).verify(csv_path, certificate_path)
    return {"status": result.status, "reason": result.reason, "run_id": result.run_id}


def build_server():
    from fastmcp import FastMCP

    server = FastMCP("SynthGauntlet")
    server.tool()(synthgauntlet_start_synthesis)
    server.tool()(synthgauntlet_get_scorecard)
    server.tool()(synthgauntlet_verify_certificate)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
