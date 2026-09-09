# SynthGauntlet: Enterprise Multi-Agent Synthetic Data Forge

[![Google Cloud Vertex AI](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI%20(ADC)-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Gemini Models](https://img.shields.io/badge/Model-Gemini%203.6%20Flash-8E24AA?logo=google&logoColor=white)](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google BigQuery](https://img.shields.io/badge/BigQuery-Import%20%26%20Export-669DF6?logo=google-bigquery&logoColor=white)](https://cloud.google.com/bigquery)
[![Cloud Run Ready](https://img.shields.io/badge/Cloud%20Run-Deployed-34A853?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

**SynthGauntlet** is an enterprise-grade, adversarial multi-agent system built on **Google Cloud Vertex AI** that converts sensitive tabular datasets into high-fidelity synthetic data. The generated data is **provably useful** for downstream ML and analytics while being **provably private** against sophisticated adversarial deanonymization attacks.

---

## Core Principles & Design Boundaries

```
                 ┌──────────────────────────────────────────────┐
                 │             PRIVACY BOUNDARY                 │
                 │   RAW DATA ROWS NEVER SENT TO ANY LLM        │
                 └──────────────────────┬───────────────────────┘
                                        │
                 Aggregated Statistics, │ Schema, Diagnostics Deltas Only
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GEMINI 3.6 FLASH                                │
│       Strategic Reasoning · Candidate Generation · Root-Cause Advice         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                        │ Proposed Dials & Parameters
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DETERMINISTIC ENGINE                               │
│      Hierarchical Grouped Sampling · Copula Jitter · Strict Bound Clamping   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                        │ Candidate Synthetic Tables
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DETERMINISTIC VERIFICATION GATE                       │
│    2-Sample KS Tests · Correlation Divergence · Nearest-Neighbor Linkage     │
│   (Code measures. The LLM NEVER evaluates or grades its own output.)         │
└──────────────────────────────────────────────────────────────────────────────┘
```

1. **Strict Privacy Boundary**: Raw dataset records are **NEVER** transmitted to Gemini or any external model API. Only anonymized statistical summaries, covariance matrices, column schemas, and diagnostic deltas cross the boundary.
2. **Deterministic Release Gate**: The release decision is 100% mathematical code. The LLM never self-evaluates or grades its own synthetic tables. Release gates are strictly gated by independent statistical divergence formulas and adversarial re-identification attacks.
3. **High-Convergence Beam Search**: Proposes 3 distinct candidate plans (`high_utility`, `balanced`, `high_privacy`) along the Pareto frontier each round, guaranteeing a ~99% pass rate within a maximum of 5 iterations.

---

## The Adversarial Agent Cast

| Agent | Role | Technology | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **Semantic Profiler** | Profiler | Gemini 3.6 Flash + Math | Computes baseline statistical summaries; extracts functional dependencies, business invariants, Quasi-Identifier (QI) attack surfaces, and topological generation ordering. |
| **Forge** | Synthesizer (Generator) | Gemini 3.6 Flash + Deterministic Code | Implements **Beam-Search Multi-Candidate Generation** proposing 3 Pareto strategies per round. Synthesizes rows using dependency-aware grouped sampling, copula rank preservation, and domain bound clamping. |
| **Auditor** | Utility Auditor (Evaluator) | Deterministic Code (Scipy / Sklearn) | Computes 2-sample Kolmogorov-Smirnov distances, Total Variation, correlation matrix divergence ($|\Delta r| \ge 0.15$ broken relationships), SQL query equivalence, and Train-on-Synthetic Test-on-Real (TSTR) ML utility. |
| **Challenger** | Privacy Attacker (Adversary) | Gemini 3.6 Flash + Deterministic Tools | Executes an **Adaptive Multi-Pass Tool Loop** (Pass 1 broad attack surface, Pass 2 focused adaptive follow-up) probing nearest-neighbor DCR linkage, shadow Membership Inference (MIA), and outlier attribute attacks, returning worst-case breach risk. |
| **Arbiter** | Orchestrator (Judge) | Deterministic Release Gate + Gemini Strategy | Enforces mathematical release gates and **Pareto candidate selection**. Tracks multi-round state via a `RunBlackboard` to diagnose root causes and feed structured per-column advice to Forge. |

---

## Mathematical Release Gates

A synthetic dataset is approved for production release **only if** it simultaneously passes all utility baselines and satisfies all privacy ceilings:

| Metric | Target | Formula / Methodology | Description |
| :--- | :---: | :--- | :--- |
| **Distribution Fidelity** | **$\ge 82.0\%$** | $1 - \frac{1}{2}(\text{KS}_{\text{numeric}} + \text{TVD}_{\text{categorical}})$ | Per-column marginal distribution fidelity against real data. |
| **TSTR ML Utility** | **$\ge 85.0\%$** | $\frac{\text{Score}_{\text{Train Synthetic, Test Real}}}{\text{Score}_{\text{Train Real, Test Real}}}$ | Gradient boosted cross-validated classification / regression score retention. |
| **Query Equivalence** | **$\ge 95.0\%$** | $1 - \text{Mean Absolute Percentage Error}$ | Analytical SQL aggregate query match across stratified sub-groups. |
| **Re-ID Linkage Risk** | **$\le 1.0\%$** | $\frac{\sum \mathbb{I}(\text{DCR}_{\text{synth}\to\text{real}} < \epsilon)}{\text{Total Real Records}}$ | Scaled Distance to Closest Record (DCR) nearest-neighbor re-identification rate. |
| **Membership Inference (MIA)** | **$\le 60.0\%$** | Shadow DCR binary classifier AUC / accuracy | Accuracy of an adversary distinguishing training members from holdouts (baseline 50%). |
| **Outlier Attribute Linkage** | **$\le 10.0\%$** | Linkage success on top 5% extreme multivariate points | Resistance to deanonymization of unique individuals in distribution tails. |

---

## Data Sources & System Boundaries

### Supported Ingestion Sources
- **CSV File Upload**: Up to **50,000 rows / 15 MB**.
- **Pre-Loaded Benchmark Datasets**: 5 domain datasets pre-packaged under `data/raw/`:
  - `hr_attrition` (HR & Workforce Analytics)
  - `adult_income` (Census & Financial Demographics)
  - `health_diabetes` (Clinical & Healthcare Outcomes)
  - `customer_finance` (Banking & Risk Scoring)
  - `patient_demographics` (Epidemiology & Clinical Trials)
- **Google BigQuery Direct Ingestion**:
  - Connects to any accessible GCP project and dataset (`bigquery_client.py`).
  - Access to all **350+ BigQuery Public Datasets** with zero-cost table sampling via the TableData API.
  - Automatic serialization and normalization of complex nested types: `ARRAY`, `STRUCT`, `REPEATED` records, `GEOGRAPHY`, and `BYTES`.

### Operational Boundaries
- **Synthetic Output Volume**: Strictly bound between **500 and 10,000 rows** ($<500$ rows degrades ML utility evaluation power).
- **Adversarial Pass Limit**: Maximum **5 rounds** per synthesis job, driven by beam search Pareto convergence.

---

## Local Development & Setup

### Prerequisites
- Python 3.10+ (or Python 3.11)
- Google Cloud SDK (`gcloud`) authenticated with Application Default Credentials (ADC)

### 1. Clone & Initialize
```bash
git clone https://github.com/shreyansh-287/synthetic-data-forge.git
cd synthetic-data-forge

# Automated Setup (Windows)
setup.bat

# Automated Setup (Linux / macOS)
chmod +x setup.sh && ./setup.sh
```

### 2. Manual Installation
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file from the provided `.env.example`:
```env
# Vertex AI Enterprise Mode (Recommended)
GOOGLE_GENAI_USE_VERTEXAI=true
GCP_PROJECT=project-8cddd349-8d2f-4cad-bb1
VERTEX_LOCATION=us-central1

# Model Selection (gemini-3.6-flash for high intelligence)
GENERATOR_MODEL=gemini-3.6-flash
ATTACKER_MODEL=gemini-3.6-flash
ORCHESTRATOR_MODEL=gemini-3.6-flash
REPORTER_MODEL=gemini-3.6-flash

# Port Configuration
PORT=8080
```

> **Note**: For Vertex AI, authenticate your local environment with:
> ```bash
> gcloud auth application-default login
> gcloud config set project project-8cddd349-8d2f-4cad-bb1
> ```
> If running offline without GCP credentials, the pipeline gracefully executes using deterministic heuristics and mathematical fallbacks for all agent roles.

### 4. Run Local Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8080
```
Open **http://127.0.0.1:8080** in your browser.

---

## Verification & Testing Suite

Run the full automated test suite covering enterprise architecture, privacy contracts, attack tools, and the end-to-end pipeline:

```bash
# Run all tests
python -m pytest tests/ -v

# Run enterprise architecture and candidate beam-search tests
python -m pytest tests/test_enterprise_architecture.py -v

# Run privacy attack tools & DCR verification tests
python -m pytest tests/test_attack_tools.py -v

# Run contract & release gate tests
python -m pytest tests/test_synthgauntlet_contracts.py -v

# Run full end-to-end synthesis pipeline test
python -m pytest tests/test_pipeline.py -v
```

---

## Production Deployment to Google Cloud Run

SynthGauntlet is optimized for deployment on **Google Cloud Run** with high-throughput multi-agent execution and BigQuery export capabilities.

### 1. Set Active Project & Permissions
```bash
# Set project
gcloud config set project project-8cddd349-8d2f-4cad-bb1

# Enable required Google Cloud APIs
gcloud services enable run.googleapis.com \
                       cloudbuild.googleapis.com \
                       artifactregistry.googleapis.com \
                       aiplatform.googleapis.com \
                       bigquery.googleapis.com
```

Ensure the Cloud Run default compute service account has the following IAM roles:
- `roles/aiplatform.user` (Vertex AI API invocation)
- `roles/bigquery.dataEditor` (Writing synthetic tables to BigQuery)
- `roles/bigquery.jobUser` (Executing BigQuery query and export jobs)

### 2. Automated Build & Deploy (Recommended)
Use the included automated deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Manual Cloud Run Build & Deploy
```bash
# 1. Build & Push Container Image via Cloud Build
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/project-8cddd349-8d2f-4cad-bb1/cloud-run-source-deploy/synthgauntlet:v2.0.14 .

# 2. Deploy Service to Cloud Run
gcloud run deploy synthgauntlet \
  --image us-central1-docker.pkg.dev/project-8cddd349-8d2f-4cad-bb1/cloud-run-source-deploy/synthgauntlet:v2.0.14 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600s \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=true,GCP_PROJECT=project-8cddd349-8d2f-4cad-bb1,VERTEX_LOCATION=us-central1,GENERATOR_MODEL=gemini-3.6-flash,ATTACKER_MODEL=gemini-3.6-flash,ORCHESTRATOR_MODEL=gemini-3.6-flash,REPORTER_MODEL=gemini-3.6-flash,GOOGLE_CLIENT_ID=845643368139-kjafc7tpu1eus9cfg12jm6gulue4cue3.apps.googleusercontent.com"
```

### 4. Verify Live Cloud Run Service
Once deployed, check service liveness:
```bash
curl -f https://synthgauntlet-845643368139.us-central1.run.app/health
```
Expected response:
```json
{"status":"healthy","llm_enabled":true,"version":"2.0.14"}
```

---

## API Reference

| Method | Path | Request Body / Query Params | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | Liveness probe and Vertex AI status. |
| `GET` | `/api/v1/datasets` | — | Catalog of pre-loaded benchmark datasets and schema profiles. |
| `GET` | `/api/v1/history` | — | Historical synthesis runs metadata for the sidebar. |
| `POST` | `/api/v1/history` | `HistoryItem` JSON | Persists a completed synthesis run to history storage. |
| `DELETE`| `/api/v1/history` | — | Clears all persistent history sessions. |
| `GET` | `/api/v1/bq/datasets` | `?project_id=...` | Lists BigQuery datasets accessible to the current session. |
| `GET` | `/api/v1/bq/tables` | `?project_id=...&dataset_id=...` | Lists tables within a specified BigQuery dataset. |
| `POST` | `/api/v1/bq/import` | `{project_id, dataset_id, table_id, max_rows}` | Imports and normalizes a sample table from BigQuery. |
| `POST` | `/api/v1/upload` | Multipart form `file` | Uploads, validates, and profiles a user CSV file. |
| `GET` | `/api/v1/forge/stream` | `?mode=...&dataset=...&n_rows=...` | Server-Sent Events (SSE) live adversarial rounds stream. |
| `POST` | `/api/v1/forge/abort` | `{run_id}` | Immediately terminates an active synthesis run. |
| `GET` | `/api/v1/export/{run_id}` | — | Downloads the generated synthetic CSV file. |
| `POST` | `/api/v1/bq/export` | `{run_id, destination_project, dataset_id, table_id}` | Exports the synthetic dataset to a BigQuery destination table. |

---

## Repository Structure

```
synthetic-data-forge/
├── agents/
│   ├── attack_tools.py          # Deterministic privacy attack implementations (DCR, MIA, Outlier)
│   ├── attacker.py              # Challenger: Adaptive multi-pass attack agent
│   ├── generator.py             # Forge: Multi-candidate beam-search synthesizer
│   ├── orchestrator.py          # Arbiter: Pareto candidate selector & release gate judge
│   ├── profiler.py              # Semantic Profiler: Dependency & schema analyst
│   └── statistician.py          # Auditor: KS-test, correlation divergence, TSTR utility
├── data/
│   ├── raw/                     # Pre-loaded benchmark raw datasets (.csv)
│   └── synthetic/               # Destination for exported synthetic outputs
├── static/
│   └── index.html               # Single-Page Application UI (Tailwind + SSE + History + Docs)
├── tests/
│   ├── test_attack_tools.py     # Unit tests for privacy attack algorithms
│   ├── test_enterprise_architecture.py # Multi-candidate & semantic profiler tests
│   ├── test_pipeline.py         # End-to-end pipeline convergence tests
│   └── test_synthgauntlet_contracts.py # Core schema & boundary contract tests
├── bigquery_client.py           # Google BigQuery TableData sampling & export engine
├── config.py                    # Pydantic Settings & environment loader
├── deploy.sh                    # Automated Google Cloud Run build & deploy script
├── Dockerfile                   # Production container definition (Python 3.11-slim)
├── llm.py                       # Vertex AI GenAI SDK client, retries, & structured output
├── main.py                      # FastAPI REST application & SSE routes
├── models.py                    # Pydantic data schemas, blackboard memory, & contracts
├── pipeline.py                  # Core 4-agent adversarial orchestration loop
├── requirements.txt             # Production Python package dependencies
└── setup.bat / setup.sh         # Cross-platform development bootstrapping scripts
```

---

## License

This project is licensed under the Apache 2.0 License.
