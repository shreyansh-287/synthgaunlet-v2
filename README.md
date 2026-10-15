# SynthGauntlet: Enterprise Multi-Agent Synthetic Data Forge

[![Google Cloud Vertex AI](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI%20(ADC)-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Gemini Models](https://img.shields.io/badge/Model-Gemini%203.6%20Flash-8E24AA?logo=google&logoColor=white)](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google BigQuery](https://img.shields.io/badge/BigQuery-Import%20%26%20Export-669DF6?logo=google-bigquery&logoColor=white)](https://cloud.google.com/bigquery)
[![Differential Privacy](https://img.shields.io/badge/Privacy-Pure%20Laplace%20%CE%B5--DP-E53935?logo=shield&logoColor=white)](#pure-differential-privacy-mode)
[![Cryptographic Audit](https://img.shields.io/badge/Audit-HMAC%20SHA--256%20Certificate-00ACC1?logo=adobe-acrobat-reader&logoColor=white)](#cryptographic-release-certificates--verification)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**SynthGauntlet** is an enterprise-grade, adversarial multi-agent system powered by **Google Cloud Vertex AI (`gemini-3.6-flash`)** that transforms sensitive tabular datasets into high-fidelity synthetic data. 

The synthesized data is **provably useful** for downstream machine learning and analytics while being **provably private** against sophisticated adversarial re-identification, linkage, and membership inference attacks.

---

## Table of Contents
1. [Core Principles & Architectural Privacy Boundary](#core-principles--architectural-privacy-boundary)
2. [Adversarial Multi-Agent Cast](#adversarial-multi-agent-cast)
3. [Synthesis Modes & Engine Strategies](#synthesis-modes--engine-strategies)
4. [PII Pre-Scan & Direct Identifier Quarantine](#pii-pre-scan--direct-identifier-quarantine)
5. [Release Policies & Adaptive Dials](#release-policies--adaptive-dials)
6. [Mathematical Release Gates & Fidelity Verification](#mathematical-release-gates--fidelity-verification)
7. [Pareto Frontier & Interactive Trade-Off Slider](#pareto-frontier--interactive-trade-off-slider)
8. [Cryptographic Release Certificates & Verification](#cryptographic-release-certificates--verification)
9. [Google BigQuery Integration & Required IAM Roles](#google-bigquery-integration--required-iam-roles)
10. [Local Development & Environment Setup](#local-development--environment-setup)
11. [Production Deployment to Google Cloud Run](#production-deployment-to-google-cloud-run)
12. [Complete REST & SSE API Reference](#complete-rest--sse-api-reference)
13. [Repository Directory Structure](#repository-directory-structure)

---

## Core Principles & Architectural Privacy Boundary

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      ENTERPRISE DATA INGESTION                              │
 │        CSV / Excel / Parquet Upload  ·  Samples  ·  Google BigQuery         │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                        Automated PII Pre-Scan & Quarantine
                        (Mask · Drop · Salted SHA-256 Hash)
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           STRICT PRIVACY BOUNDARY                           │
 │               RAW DATA ROWS ARE NEVER TRANSMITTED TO ANY LLM                │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                 Aggregated Statistics, │ Schemas, Diagnostic Deltas Only
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     GOOGLE CLOUD VERTEX AI (GEMINI 3.6 FLASH)               │
 │           Strategic Reasoning · Candidate Generation · Root-Cause Advice    │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ Proposed Dials & Synthesis Plans
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         DETERMINISTIC FORGE ENGINE                          │
 │     Hierarchical Grouped Sampling · Copula Jitter · Pure Laplace DP Engine  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ Synthetic Candidate Batches
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     DETERMINISTIC VERIFICATION GATES                        │
 │     2-Sample KS Tests · ML Utility (TSTR) · Nearest-Neighbor Linkage (DCR)  │
 │     Shadow Membership Inference (MIA) · Outlier Attribute Disclosure        │
 │              (Code measures. The LLM NEVER grades its own work.)            │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
           PASS RELEASE GATES                     FAIL RELEASE GATES
     ✅ Cryptographic PDF Certificate       🛑 Execution Aborted / Quarantine
     ✅ Direct BigQuery Warehouse Export     ⚠️ Failure Diagnostic PDF Report
     ✅ Verified Synthetic CSV Download     🔄 Optional 3-Step Auto-Heal Rescue
```

1. **Strict Zero-Row Privacy Boundary**: Raw dataset records **never** cross the network to Gemini or any external model API. Only anonymized statistical profiles, correlation matrices, column schemas, and diagnostic deltas are processed by LLM agents.
2. **Deterministic Mathematical Release Decisions**: LLMs propose candidate parameters and diagnose errors, but **never evaluate or approve their own output**. Release gating is strictly determined by independent statistical divergence metrics and adversarial re-identification attacks.
3. **Multi-Candidate Beam Search**: Evaluates 3 distinct candidate strategies (`high_utility`, `balanced`, `high_privacy`) along the Pareto frontier per round, converging within bounded iterations.

---

## Adversarial Multi-Agent Cast

| Agent | Icon | Engine | Core Responsibility |
| :--- | :---: | :--- | :--- |
| **Semantic Profiler** | 📋 | Gemini 3.6 Flash + Code | Computes baseline statistical profiles, infers column domains, extracts functional dependencies, Quasi-Identifier (QI) attack surfaces, sensitive attributes, and generation topological order. |
| **Forge** | ⚗️ | Gemini 3.6 Flash + Deterministic Code | Generates 3 Pareto candidates per round using dependency-aware grouped sampling, copula rank preservation, and domain-bound clamping. |
| **Auditor** | 📏 | Deterministic Code (Scipy / Sklearn) | Evaluates statistical distribution divergence (Kolmogorov-Smirnov + Total Variation), correlation matrix shift ($|\Delta r| \ge 0.15$), SQL aggregate query equivalence, and Train-on-Synthetic Test-on-Real (TSTR) ML utility. |
| **Challenger** | 🕵️ | Gemini 3.6 Flash + Deterministic Tools | Conducts multi-pass adversarial attacks probing Distance to Closest Record (DCR) nearest-neighbor linkage, shadow Membership Inference (MIA), and outlier attribute disclosure. |
| **Arbiter** | ⚖️ | Deterministic Gates + Gemini Reasoning | Evaluates release gates, marks Pareto non-dominated candidates, selects the winning table, and orchestrates the Auto-Heal rescue ladder when gates fail. |
| **DP Engine** | 🛡️ | Pure Mathematical Code (`dp_engine.py`) | Provides formal mathematical Differential Privacy via the calibrated Laplace mechanism, tracking expended $\varepsilon$ privacy budget. |

---

## Synthesis Modes & Engine Strategies

SynthGauntlet supports three distinct synthesis pathways tailored to enterprise requirements:

### 1. Hybrid Mode (Recommended Standard)
- Combines Gaussian Copula rank preservation, dependency-aware hierarchical grouped sampling, and multi-candidate beam search.
- Maximizes downstream machine learning accuracy (TSTR $\ge 85\%$) while guaranteeing adversarial empirical safety.

### 2. Pure Differential Privacy ($\varepsilon$-DP) Mode
- Generates synthetic data with **formal mathematical Differential Privacy** guarantees.
- Bounded columns are perturbed using calibrated **Laplace noise** scaled to query sensitivity $\Delta f / \varepsilon$.
- Users configure an explicit privacy budget $\varepsilon \in [0.1, 10.0]$ (Default: $\varepsilon = 1.0$).
- Output reports exact spent $\varepsilon$ budget and formal mathematical protection text.

### 3. Auto-Heal Rescue Mode
When a candidate narrowly misses the privacy ceiling or utility floor, the Arbiter can activate an automated **3-Step Rescue Ladder**:
1. **Tighten**: Increases recombination shuffle and noise while raising rare category frequency thresholds.
2. **Generalize**: Applies targeted QI bucketization, quantile binning, and low-DCR row dropping.
3. **Second Generation**: Generates fresh candidates across 4 distinct random seeds using repaired generative blueprints.

---

## PII Pre-Scan & Direct Identifier Quarantine

Before synthesis begins, SynthGauntlet includes an automated **PII Pre-Scan** (`pii_scanner.py`):
- **Detected Entities**: Email addresses, US/International phone numbers, Social Security Numbers (SSN), Credit Card numbers, IP addresses, Physical street addresses, and Full names.
- **Quarantine Actions**:
  - **Mask**: Replaces values with standardized tokens (e.g., `user***@domain.com`, `XXX-XX-1234`).
  - **Drop**: Completely removes the column before statistical profiling.
  - **Salted SHA-256 Hash**: Replaces identifiers with irreversible cryptographic hashes for joinable entity resolution without data leakage.

---

## Release Policies & Adaptive Dials

SynthGauntlet supports 3 preset policies plus **Natural Language Policy Interpretation**:

| Policy | Target Audience | Fidelity Floor | TSTR ML Floor | Linkage Ceiling | MIA Ceiling |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Balanced** | General Analytics & ML | $\ge 82.0\%$ | $\ge 85.0\%$ | $\le 1.0\%$ | $\le 60.0\%$ |
| **Privacy First** | Regulated Healthcare & Finance | $\ge 75.0\%$ | $\ge 78.0\%$ | $\le 0.5\%$ | $\le 55.0\%$ |
| **Analytics First** | Internal Research & Benchmarks | $\ge 90.0\%$ | $\ge 88.0\%$ | $\le 2.0\%$ | $\le 65.0\%$ |

- **Gemini Policy Translator (`/api/v1/policy/interpret`)**: Enter freeform compliance instructions (e.g., *"We need HIPAA-compliant clinical data for an external research challenge"*). Gemini translates the instruction into deterministic mathematical threshold numbers and explains the trade-offs.

---

## Mathematical Release Gates & Fidelity Verification

A synthetic dataset is approved for production release **only if** it simultaneously passes all utility floors and stays beneath all privacy ceilings:

| Release Gate | Standard Target | Formula / Methodology | Description |
| :--- | :---: | :--- | :--- |
| **Distribution Fidelity** | **$\ge 82.0\%$** | $1 - \frac{1}{2}(\text{KS}_{\text{numeric}} + \text{TVD}_{\text{categorical}})$ | Per-column marginal probability fidelity against original data. |
| **TSTR ML Utility** | **$\ge 85.0\%$** | $\frac{\text{Score}_{\text{Train Synthetic, Test Real}}}{\text{Score}_{\text{Train Real, Test Real}}}$ | Gradient boosted cross-validated classification / regression retention. |
| **Query Equivalence** | **$\ge 95.0\%$** | $1 - \text{Mean Absolute Percentage Error}$ | Analytical SQL aggregate query parity across stratified subgroups. |
| **Re-ID Linkage Risk** | **$\le 1.0\%$** | $\frac{\sum \mathbb{I}(\text{DCR}_{\text{synth}\to\text{real}} < \epsilon)}{\text{Total Real Records}}$ | Scaled Distance to Closest Record (DCR) nearest-neighbor attack. |
| **Membership Inference (MIA)** | **$\le 60.0\%$** | Shadow DCR binary classifier AUC | Accuracy of an adversary detecting training members vs holdouts (baseline 50%). |
| **Outlier Attribute Linkage** | **$\le 10.0\%$** | Linkage success on top 5% multivariate tail | Resistance to deanonymization of unique individuals in distribution tails. |
| **Attribute Inference Advantage** | **$\le 25.0\%$** | Supervised target prediction gain over mode | Probes if adversary can infer sensitive attributes knowing Quasi-Identifiers. |
| **Auxiliary Linkage** | **$\le 15.0\%$** | Cross-join re-identification rate | Resistance against external secondary dataset correlation and joins. |
| **Near-Duplicate Rows** | **$\le 10.0\%$** | $\frac{\text{Near Duplicates}}{\text{Total Synthetic Records}}$ | Identifies synthetic rows that too closely mirror training data (memorization ceiling). |

### Visual Verification: Column Distribution Overlay (Fidelity Inspector)
Directly integrated into the Dual Scorecard UI, the **Column Distribution Overlay** allows data practitioners to visually verify empirical alignment:
- **Continuous Features**: 10 normalized histogram buckets across unified bounds with real vs. synthetic comparative percentage bars, Mean, Standard Deviation, and `[min, max]` range.
- **Categorical Features**: Top categories with relative row frequencies (`% of total rows`), cardinality tracking, and tail grouping into an "Other" bucket.
- **Interactive Selector**: Instant column switching with percentage delta ($\Delta\%$) badges to confirm marginal distribution retention without data memorization.

---

## Pareto Frontier & Interactive Trade-Off Slider

SynthGauntlet plots every candidate evaluated across all adversarial rounds onto an interactive **Privacy–Utility Pareto Frontier**:
- **Non-Dominated Candidates**: Highlights optimal candidate solutions where utility cannot be improved without increasing privacy risk.
- **Interactive Trade-Off Slider**: Allows data stewards to smoothly slide between **Max Privacy** and **Max Utility** among certified candidates.
- **Instant Table Switching (`/api/v1/forge/{run_id}/active-candidate`)**: Dynamically updates the active release table behind CSV downloads and BigQuery exports without re-running synthesis.

---

## Cryptographic Release Certificates & Verification

Every approved dataset is sealed with an automated cryptographic audit trail:
- **Branded 2-Page PDF Certificate (`/api/v1/certificate/{run_id}/pdf`)**:
  - Displays all 6 release gate scores, final status, difficulty breakdown, and run metadata.
  - Contains **SHA-256 Data Hash** of the training split and final synthetic table.
  - Cryptographically signed with a secure **HMAC-SHA256 Audit Signature**.
- **Tamper-Evident Verification (`/api/v1/verify`)**:
  - Users can upload any synthetic CSV alongside its PDF Certificate to cryptographically verify data integrity and confirm that records were not tampered with post-release.
- **Failure Diagnostics Report (`/api/v1/failure-report/{run_id}/pdf`)**:
  - If release gates fail, generates an uncertified PDF report diagnosing exact attack breach vectors and remedial action recommendations.

---

## Google BigQuery Integration & Required IAM Roles

SynthGauntlet integrates directly with Google Cloud BigQuery for frictionless enterprise cloud pipelines:

### 1. Ingestion / Import
- Pull tables directly from any accessible GCP project and dataset.
- Supports **350+ BigQuery Public Datasets** with zero-cost table sampling via the TableData API.
- Automatically unpacks complex nested types (`ARRAY`, `STRUCT`, `REPEATED` records, `GEOGRAPHY`).
- **Required IAM Roles for Import**:
  - Public Datasets: Zero credentials required.
  - Private Enterprise Tables: `roles/bigquery.dataViewer` + `roles/bigquery.jobUser`.

### 2. Export / Publish
- One-click export of certified synthetic data directly into your BigQuery data warehouse.
- Automatically sanitizes column names to comply with BigQuery naming specifications.
- **Required IAM Roles for Export**:
  - `roles/bigquery.dataEditor` (Dataset & Table Creation/Write) + `roles/bigquery.jobUser` (Job Execution).

---

## Local Development & Environment Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Google Cloud SDK (`gcloud`) authenticated with Application Default Credentials (ADC)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/shreyansh-287/synthgauntlet-v2.git
cd synthgauntlet-v2

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (`.env`)
Create a `.env` file in the project root:
```env
# Google Cloud Vertex AI Configuration
LLM_PROVIDER=gemini
GOOGLE_GENAI_USE_VERTEXAI=true
GCP_PROJECT=project-8cddd349-8d2f-4cad-bb1
VERTEX_LOCATION=global

# Model Selection (gemini-3.6-flash recommended for high speed & intelligence)
GENERATOR_MODEL=gemini-3.6-flash
ATTACKER_MODEL=gemini-3.6-flash
ORCHESTRATOR_MODEL=gemini-3.6-flash
REPORTER_MODEL=gemini-3.6-flash

# Port & Server Settings
PORT=8080
```

> **Note on GCP Authentication**:
> Authenticate your local development machine with Google Cloud ADC:
> ```bash
> gcloud auth application-default login
> gcloud config set project project-8cddd349-8d2f-4cad-bb1
> ```
> *If offline without GCP credentials, the pipeline gracefully falls back to deterministic heuristics and mathematical baselines.*

### 3. Start Local Development Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8080
```
Open **http://127.0.0.1:8080** in your browser.

### 4. Running the Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run distribution overlay unit tests
python -m pytest tests/test_distribution.py -v

# Run Pareto frontier trade-off tests
python -m pytest tests/test_pareto.py -v

# Run Auto-Heal rescue ladder tests
python -m pytest tests/test_auto_heal.py -v

# Run BigQuery authentication and export tests
python -m pytest tests/test_bq_auth.py tests/test_bq_export.py -v
```

---

## Production Deployment to Google Cloud Run

SynthGauntlet is built container-first and ready for automated deployment to **Google Cloud Run**:

```bash
# 1. Enable required GCP services
gcloud services enable run.googleapis.com \
                       cloudbuild.googleapis.com \
                       artifactregistry.googleapis.com \
                       aiplatform.googleapis.com \
                       bigquery.googleapis.com

# 2. Deploy using included deployment script
chmod +x deploy.sh
./deploy.sh
```

---

## Complete REST & SSE API Reference

| Method | Endpoint | Request Body / Query | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | Service health check, Vertex AI connectivity & token counters. |
| `GET` | `/api/v1/tokens` | — | Real-time LLM token usage and call telemetry. |
| `GET` | `/api/v1/datasets` | — | Catalog of pre-loaded benchmark sample datasets. |
| `GET` | `/api/v1/sample/profile` | `?dataset=...` | Schema and statistical profile of a benchmark sample. |
| `POST` | `/api/v1/upload` | Multipart `file` | Uploads, validates, and profiles a user CSV/tabular file. |
| `POST` | `/api/v1/scan-pii` | `{upload_id}` | Scans uploaded dataset for direct PII identifiers. |
| `POST` | `/api/v1/apply-pii` | `{upload_id, actions}` | Applies Mask, Drop, or Salted Hash actions to PII columns. |
| `GET` | `/api/v1/policy/presets` | — | Retrieves preset release policies and thresholds. |
| `POST` | `/api/v1/policy/interpret` | `{policy_text}` | Gemini-powered natural language policy translation. |
| `POST` | `/api/v1/start-run` | `ForgeRequest` JSON | Initiates asynchronous multi-agent adversarial synthesis run. |
| `GET` | `/api/v1/forge/stream` | `?run_id=...&n_rows=...` | Server-Sent Events (SSE) real-time round-by-round stream. |
| `POST` | `/api/v1/forge/abort` | `{run_id}` | Immediately terminates an active synthesis run. |
| `GET` | `/api/v1/status/{run_id}` | — | Polls run status, agent progress events, and partial scores. |
| `GET` | `/api/v1/runs/{run_id}` | — | Retrieves full persisted `ForgeResponse` JSON result. |
| `POST` | `/api/v1/runs/{run_id}` | `ForgeResponse` JSON | Persists full run result to local storage. |
| `GET` | `/api/v1/history` | — | Lists historical synthesis sessions for sidebar restoration. |
| `POST` | `/api/v1/forge/{run_id}/active-candidate`| `{candidate_id}` | Switches active release candidate along the Pareto frontier. |
| `GET` | `/api/v1/export/{run_id}` | `?format=csv&candidate_id=...` | Downloads certified synthetic CSV file. |
| `GET` | `/api/v1/certificate/{run_id}/pdf` | — | Downloads 2-page cryptographic PDF release certificate. |
| `POST` | `/api/v1/verify` | Multipart `csv_file` + `pdf_file` | Cryptographically verifies synthetic data against certificate HMAC. |
| `GET` | `/api/v1/failure-report/{run_id}/pdf` | — | Downloads uncertified failure diagnostics report PDF. |
| `GET` | `/api/v1/bq/datasets` | `?project_id=...` | Lists BigQuery datasets accessible via user OAuth token. |
| `GET` | `/api/v1/bq/tables` | `?project_id=...&dataset_id=...` | Lists tables within a specified BigQuery dataset. |
| `POST` | `/api/v1/bq/import` | `{project_id, dataset_id, table_id}` | Imports sample table from BigQuery into SynthGauntlet. |
| `POST` | `/api/v1/bq/export` | `{run_id, project_id, dataset_id, table_id}` | Directly exports certified synthetic data to a BigQuery table. |

---

## Repository Directory Structure

```
synthgauntlet-v2/
├── agents/
│   ├── attack_tools.py          # Deterministic privacy attack implementations (DCR, MIA, Outlier)
│   ├── attacker.py              # Challenger: Adaptive multi-pass privacy attack agent
│   ├── dp_engine.py             # Pure Laplace Differential Privacy engine & epsilon accounting
│   ├── generator.py             # Forge: Multi-candidate beam-search tabular synthesizer
│   ├── orchestrator.py          # Arbiter: Pareto candidate selector, release gate & Auto-Heal
│   ├── profiler.py              # Semantic Profiler: Dependency, schema & attack surface analyst
│   └── statistician.py          # Auditor: KS-test, correlation divergence, TSTR ML benchmark
├── data/
│   ├── raw/                     # Pre-loaded benchmark raw datasets (Adult, Diabetes, etc.)
│   ├── run_results/             # Persistent JSON storage for historical runs
│   └── output/                  # Temporary cache for exported synthetic CSVs
├── static/
│   ├── index.html               # Single-Page Application UI (Tailwind, Chart.js, SSE, Modals)
│   ├── synthgauntlet-logo.png   # Application branding asset
│   └── synthgauntlet-banner-bg.png
├── tests/
│   ├── test_ai_usage.py         # AI usage tracking & token accounting tests
│   ├── test_attack_tools.py     # Unit tests for privacy attack algorithms
│   ├── test_auto_heal.py        # Automated rescue ladder & candidate repair tests
│   ├── test_bq_auth.py          # BigQuery OAuth & token handling tests
│   ├── test_bq_export.py        # BigQuery table export & column sanitization tests
│   ├── test_distribution.py     # Column Distribution Overlay & histogram tests
│   ├── test_dp_engine.py        # Pure Laplace DP mechanism tests
│   ├── test_pareto.py           # Pareto frontier candidate switching & skyline tests
│   ├── test_pipeline.py         # End-to-end pipeline convergence tests
│   └── test_synthgauntlet_contracts.py # Data contracts, bounds & schema tests
├── bigquery_client.py           # Google BigQuery TableData API & export client
├── certificate.py               # Branded 2-page cryptographic PDF audit certificate generator
├── failure_report.py            # Uncertified failure diagnostics PDF report generator
├── pii_scanner.py               # Direct PII entity scanner (Mask, Drop, Salted Hash)
├── reporter.py                  # Dual Scorecard builder, Pareto calculations & CSV exporter
├── data_source.py               # Ingestion abstraction layer (CSV, Samples, BigQuery)
├── config.py                    # Pydantic Settings & environment variable configuration
├── llm.py                       # Vertex AI GenAI SDK client, AFC, tokens & timeout handling
├── models.py                    # Pydantic V2 data contracts, schemas & Blackboard models
├── pipeline.py                  # Core adversarial orchestration loop & distribution engine
├── main.py                      # FastAPI REST application, SSE routes & lifecycle handlers
├── requirements.txt             # Production Python dependencies
├── Dockerfile                   # Cloud Run container definition (Python 3.11-slim)
└── deploy.sh                    # Automated Google Cloud Run build & deploy script
```

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
