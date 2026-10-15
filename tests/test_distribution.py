import numpy as np
import pandas as pd
from models import ForgeResponse
from pipeline import compute_column_distributions


def test_compute_column_distributions_numeric_and_categorical():
    np.random.seed(42)
    real_df = pd.DataFrame({
        "age": np.random.normal(40, 10, 200),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], 200),
        "income": np.random.exponential(50000, 200),
    })
    synth_df = pd.DataFrame({
        "age": np.random.normal(39.5, 9.8, 200),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], 200),
        "income": np.random.exponential(49000, 200),
    })

    res = compute_column_distributions(real_df, synth_df)
    assert res is not None
    assert "columns" in res
    assert "age" in res["columns"]
    assert "department" in res["columns"]

    age_info = res["details"]["age"]
    assert age_info["type"] == "numeric"
    assert len(age_info["labels"]) == 10
    assert len(age_info["real_pct"]) == 10
    assert len(age_info["synth_pct"]) == 10
    assert "mean" in age_info["real_stats"]
    assert "std" in age_info["synth_stats"]

    dept_info = res["details"]["department"]
    assert dept_info["type"] == "categorical"
    assert len(dept_info["labels"]) >= 4
    assert "top" in dept_info["real_stats"]
    assert "unique" in dept_info["synth_stats"]


def test_compute_column_distributions_graceful_failures():
    assert compute_column_distributions(None, None) is None
    assert compute_column_distributions(pd.DataFrame(), pd.DataFrame()) is None

    # Empty column
    df1 = pd.DataFrame({"empty_col": [np.nan, np.nan]})
    df2 = pd.DataFrame({"empty_col": [np.nan, np.nan]})
    assert compute_column_distributions(df1, df2) is None


def test_forge_response_column_distributions_serialization():
    resp = ForgeResponse(
        run_id="test-run",
        status="SUCCESS",
        column_distributions={"columns": ["age"], "details": {"age": {"type": "numeric"}}},
    )
    dumped = resp.model_dump()
    assert dumped["column_distributions"] is not None
    assert dumped["column_distributions"]["columns"] == ["age"]
