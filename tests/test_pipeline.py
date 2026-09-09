"""End-to-end pipeline test on the static CSVs (runs in deterministic mode)."""

import pytest

from constants import DATASETS
from models import ForgeRequest, SourceSpec
from pipeline import run_forge


@pytest.mark.parametrize("dataset_key", list(DATASETS.keys()))
def test_forge_runs_on_every_static_dataset(dataset_key):
    if not DATASETS[dataset_key]["path"].exists():
        pytest.skip(f"{dataset_key} CSV missing")
    req = ForgeRequest(
        source=SourceSpec(type="csv", dataset_key=dataset_key),
        n_rows=600,
    )
    resp = run_forge(req)

    assert resp.status in {"SUCCESS", "FAILED_TO_CONVERGE"}
    assert len(resp.rounds) >= 1
    assert resp.export_path is not None
    assert resp.scorecard is not None
    assert "DUAL SCORECARD" in resp.scorecard_summary

    final = resp.rounds[-1]
    assert 0.0 <= final.utility.fidelity_score <= 1.0
    assert 0.0 <= final.attack.re_id_rate <= 1.0
    assert 0.0 <= final.attack.membership_accuracy <= 1.0


def test_successful_run_exports_a_dual_scorecard():
    if not DATASETS["adult_income"]["path"].exists():
        pytest.skip("adult_income CSV missing")
    response = run_forge(
        ForgeRequest(
            source=SourceSpec(type="csv", dataset_key="adult_income"),
            n_rows=300,
        )
    )
    assert response.status == "SUCCESS"
    assert response.export_path is not None
    assert response.scorecard is not None
    assert response.scorecard.passed


def test_abort_run_generates_zero_data():
    from pipeline import abort_run
    run_id = "test_abort_123"
    abort_run(run_id)
    req = ForgeRequest(
        source=SourceSpec(type="csv", dataset_key="health_diabetes"),
        n_rows=300,
    )
    resp = run_forge(req, run_id=run_id)
    assert resp.status == "ABORTED"
    assert resp.export_path is None

