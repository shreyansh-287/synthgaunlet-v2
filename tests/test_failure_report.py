"""Failure-report PDF: exact gate failures for Laplace and hybrid runs."""

import json

from failure_report import issue, load_report, render_pdf


def _response(**overrides):
    payload = {
        "status": "FAILED_TO_CONVERGE",
        "final_status": "NOT_CERTIFIED",
        "dataset_key": "health_diabetes",
        "domain": "health",
        "policy_name": "Balanced",
        "policy_explanation": "Default balanced thresholds.",
        "formal_dp_guarantee": "No formal DP guarantee",
        "auto_heal_enabled": True,
        "thresholds": {
            "fidelity_min": 0.82,
            "tstr_min": 0.85,
            "query_equivalence_min": 0.95,
            "reid_max": 0.01,
            "membership_max": 0.60,
            "attribute_inference_max": 0.25,
            "linkage_max": 0.15,
            "near_duplicate_max": 0.10,
        },
        "rounds": [{
            "round_index": 1,
            "passed": False,
            "plan": {"plan_name": "balanced"},
            "utility": {
                "fidelity_score": 0.91,
                "utility_ratio": 0.97,
                "query_equivalence": 0.71,
                "tstr_detail": {"task": "classification", "acc_real": 0.74, "acc_synth": 0.70, "top_features": ["glucose"]},
                "diagnostic_report": {"weakest_columns": ["glucose"], "broken_relationships": []},
                "fairness": {"status": "PASS", "summary": "No demographic attributes detected.", "groups": []},
            },
            "attack": {
                "re_id_rate": 0.04,
                "membership_accuracy": 0.51,
                "attribute_inference_advantage": 0.0,
                "linkage_rate": 0.0,
                "near_duplicate_rate": 0.0,
            },
        }],
        "rescue_attempts": [],
    }
    payload.update(overrides)
    return payload


def test_hybrid_failure_report_names_the_failed_gates_and_engine():
    report = issue("fail-hybrid", _response(), {"synthesis_mode": "hybrid", "n_rows": 300})
    assert report["engine_summary"].startswith("This run used the hybrid marginal engine.")
    failed = [item["label"] for item in report["findings"] if item["outcome"] == "FAIL"]
    assert failed == ["Query equivalence", "Re-identification risk"]
    stored = load_report("fail-hybrid")
    assert stored["run_id"] == "fail-hybrid"
    pdf = render_pdf(stored)
    assert pdf.startswith(b"%PDF")
    assert b"SYNTHGAUNTLET-CERTIFICATE" not in pdf


def test_laplace_failure_report_records_the_spent_budget():
    report = issue(
        "fail-laplace",
        _response(formal_dp_guarantee="Formal DP guarantee: epsilon = 1 (Laplace mechanism)"),
        {
            "synthesis_mode": "dp",
            "dp_epsilon": 1.0,
            "epsilon_spent": 1.0,
            "dp_queries": ["1-way:glucose", "2-way:glucose|outcome"],
            "n_rows": 300,
        },
    )
    assert "Laplace" in report["engine_summary"]
    assert "epsilon 1.0000 of 1.0000" in report["engine_summary"]
    assert "2-way:glucose|outcome" in report["engine_summary"]
    document = json.dumps(report)
    assert "Formal DP guarantee" in document
    assert render_pdf(report).startswith(b"%PDF")
