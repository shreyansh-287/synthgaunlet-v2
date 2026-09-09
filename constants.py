"""
Central registry of the static demo datasets for Synthetic Data Forge.

The CSVs are committed under data/raw/ and treated as static input -- there is
no download step. Source links are kept only for provenance / re-fetching by
hand if ever needed.

Metadata fields per dataset:
  domain              -> demo domain (health | finance | hr)
  path                -> local CSV path (absolute)
  source_page         -> human-facing landing page / provenance
  source_url          -> original direct link (reference only)
  license             -> usage terms
  quasi_identifiers   -> columns an attacker could use for re-identification
  sensitive_columns   -> the "secret" the attacker tries to steal
  label_column        -> target column for the TSTR utility proof
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"

DATASETS = {
    "adult_income": {
        "domain": "finance",
        "path": DATA_DIR / "adult_income.csv",
        "source_page": "https://archive.ics.uci.edu/dataset/2/adult",
        "source_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "license": "CC BY 4.0 (UCI Machine Learning Repository)",
        "quasi_identifiers": ["age", "sex", "race", "native_country", "marital_status"],
        "sensitive_columns": ["income"],
        "label_column": "income",
        "display_name": "Bank Customers",
        "rows": 32561,
        "columns": 15,
        "description": "US Census demographic details for prediction task.",
    },
    "health_diabetes": {
        "domain": "health",
        "path": DATA_DIR / "health_diabetes.csv",
        "source_page": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "source_url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "license": "CC0 Public Domain (originally UCI / NIDDK)",
        "quasi_identifiers": ["age", "pregnancies"],
        "sensitive_columns": ["outcome"],
        "label_column": "outcome",
        "display_name": "Health Diabetes",
        "rows": 768,
        "columns": 9,
        "description": "Patient stats for diabetes onset prediction.",
    },
    "hr_attrition": {
        "domain": "hr",
        "path": DATA_DIR / "hr_attrition.csv",
        "source_page": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "source_url": "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
        "license": "Database Contents License (IBM sample data)",
        "quasi_identifiers": ["Age", "Gender", "MaritalStatus", "Department"],
        "sensitive_columns": ["MonthlyIncome", "Attrition"],
        "label_column": "Attrition",
        "display_name": "HR Employee Attrition",
        "rows": 1470,
        "columns": 35,
        "description": "Company staff attrition & roles details.",
    },
}


# ============================================================================
# Gemini credentials are supplied through the environment or Secret Manager.
# Never commit a real key to source control.
# ============================================================================
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = ""