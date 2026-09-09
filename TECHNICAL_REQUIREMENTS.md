# Technical Requirements Document (TRD): Synthetic Data Forge

**Version:** 1.0.0
**Runtime:** Python 3.11
**Target Platform:** Google Cloud Run (Serverless Container)
**Primary GCP Services:** Vertex AI (Gemini 2.x), BigQuery, Cloud Run, Artifact Registry, Cloud Logging
**Architecture Pattern:** Adversarial Multi-Agent Loop with Deterministic Verification

---

## 1. System Context

Synthetic Data Forge is a cloud-native microservice that transforms a sensitive tabular dataset into a synthetic equivalent that is **statistically faithful** and **provably resistant to privacy attacks**. It combines LLM-driven agents (generation, adversarial attack strategy, orchestration) with deterministic statistical engines (profiling, utility scoring, attack execution).

**Design principle:** *Code measures. The LLM decides and creates.* Every metric is computed deterministically; the LLM never grades its own output.

---

## 2. High-Level Architecture

```
                         +-----------------------------+
                         |   Client / Front-End (UI)   |
                         |  (Streamlit / React + API)  |
                         +--------------+--------------+
                                        |
                             HTTP POST /api/v1/forge
                                        |
                         +--------------v--------------+
                         |    GCP Cloud Run Service     |
                         |     (FastAPI App Context)    |
                         +--------------+--------------+
                                        |
+---------------------------------------v-----------------------------------------+
| Adversarial Multi-Agent Orchestration Layer                                     |
|                                                                                 |
|   [CODE] Profiler ---------> [LLM] Generator Agent                              |
|      (stats only)                    |                                          |
|                                      v                                          |
|                          [CODE] Statistician (Utility)                          |
|                                      |                                          |
|                                      v                                          |
|                    [LLM AGENT] Attacker (reasons + calls tools)                 |
|             tools: re-identification | membership inference | outlier linkage   |
|                                      |                                          |
|                             [CODE] Decision Gate                                |
|                             /                  \                                |
|                        (Pass)                  (Fail)                            |
|                        /                          \                             |
|          +------------v-----------+     +-----------v------------------+         |
|          | [LLM] Report Writer    |     | [LLM] Orchestrator Strategy  |         |
|          | Publish to BigQuery    |     | -> feedback to Generator     |---loop  |
|          +------------------------+     +------------------------------+         |
+---------------------------------------------------------------------------------+
```

---

## 3. Agent & Component Specifications

### 3.1 Statistical Profiler (`profiler.py`) — CODE
- **Role:** Extract a statistical fingerprint of the source data; **never expose raw rows to the LLM**.
- **Outputs:** per-column type, distribution summary (mean/std/min/max/quantiles for numeric; value frequencies for categorical), pairwise correlation matrix, detected conditional rules (e.g., value co-occurrence).
- **Libraries:** `pandas`, `numpy`.

### 3.2 Generator Agent (`generator.py`) — LLM
- **Role:** Produce synthetic rows matching the profile + optional business rules + Orchestrator feedback.
- **Model:** Gemini 2.x (Flash for speed / Pro for quality) via Vertex AI SDK.
- **Output enforcement:** Structured output via Pydantic `response_schema` (JSON rows).
- **Input:** profile JSON, business rules, prior-round correction instructions.
- **Batching:** generate in chunks (e.g., 500 rows/call) to stay within token limits.

### 3.3 Statistician / Utility Agent (`statistician.py`) — CODE
- **Role:** **Prove** the synthetic data is genuinely useful, not just superficially similar. Climbs a utility hierarchy from weak (distributions) to strong (task performance).
- **Metrics (weakest → strongest):**
  - **L1 — Distribution fidelity:** Kolmogorov-Smirnov (`scipy.stats.ks_2samp`) for numeric; total-variation distance for categorical.
  - **L2 — Correlation preservation:** mean-abs / Frobenius delta between the real and synthetic correlation matrices.
  - **L3 — Query equivalence:** run identical analytical SQL (GROUP BY aggregates: AVG/SUM/COUNT) on real vs synthetic; report per-aggregate % match. Great for the live demo.
  - **L4 — TSTR ("Train on Synthetic, Test on Real") — the headline proof:**
    - **TRTR baseline:** train a model on REAL data, test on held-out REAL data → `acc_real`.
    - **TSTR:** train the *same* model on SYNTHETIC data, test on the *same* held-out REAL data → `acc_synth`.
    - **`utility_ratio = acc_synth / acc_real`** — e.g., 0.83 / 0.87 = 0.95 → "works 95% as well as real data."
    - Implementation: `scikit-learn` (portable) or **BigQuery ML** (`CREATE MODEL` on synthetic + `ML.EVALUATE` on real) as a GCP-native flex.
- **Output:** `fidelity_score`, `correlation_delta`, `query_equivalence`, `utility_ratio` (TSTR) → combined into the Utility Scorecard.

### 3.4 Attacker Agent (`attacker.py`) — LLM AGENT (reasoning + tool use)
- **Role:** An autonomous adversarial agent whose sole objective is to **break the privacy of the synthetic data**. It is genuinely LLM-driven: it *reasons* about where the data is weakest, *plans* an attack, *invokes deterministic attack tools*, *interprets* the numeric results, and *adapts* its plan across rounds. It does not run a fixed script.
- **Model:** Gemini 2.x via Vertex AI SDK, using function/tool calling with structured output.
- **Agent loop (per round):**
  1. **Observe** — receives the dataset profile, the flagged examples + metrics from prior rounds, and the Generator's last correction. (It never sees raw source rows — only the synthetic data and attack tool outputs.)
  2. **Hypothesize** — reasons in natural language about the most likely privacy weakness. *Example:* "Last round ZIP was generalized, so linkage there is dead. But `diagnosis=Pregnancy` only occurs for young females — that rare combo is probably still re-identifiable. I'll target the age+diagnosis subspace."
  3. **Plan** — decides which tools to call, on which columns/subsets, and with what parameters (e.g., which quasi-identifiers to weight, distance threshold to probe, which subgroup to attack).
  4. **Act (tool calls)** — invokes the deterministic attack tools below. The agent *chooses and parameterizes* them; the tools do the math so results are trustworthy.
  5. **Interpret & Adapt** — reads the returned numbers, updates its belief about weaknesses, and either escalates a promising attack or pivots. Produces a natural-language `attacker_strategy` explaining what it tried and why.
- **Deterministic attack tools the agent can call (CODE — pure math, no LLM):**
  - `run_reidentification(quasi_identifiers, threshold, subset=None)` — encodes rows numerically; for each synthetic row finds its nearest real neighbor (`sklearn.neighbors.NearestNeighbors`); flags if distance < threshold; returns `re_id_rate = flagged / total` + flagged examples.
  - `run_membership_inference(features)` — splits real data into IN (used to build the profile) and OUT (held out); computes each record's distance to the nearest synthetic row; trains a classifier (`sklearn`) on IN/OUT labels; returns test accuracy (≈ 0.5 = safe).
  - `run_outlier_linkage(columns)` — targets rare/unique value combinations (highest re-identification risk) specifically.
- **Why this is a true agent, not a scorer:** the deterministic tools cannot decide *where* or *how* to attack — the LLM does, and its strategy demonstrably changes in response to the Generator's defenses each round. This is the adversarial half of the multi-agent duel.
- **Output:** `re_id_rate`, `membership_accuracy`, `flagged_examples`, and `attacker_strategy` (the agent's reasoning + chosen plan).

### 3.5 Decision Gate (`orchestrator.py`) — CODE
- **Logic:** `pass = fidelity >= FIDELITY_MIN and re_id_rate <= REID_MAX and membership_accuracy <= MEMBERSHIP_MAX`.
- Defaults: `FIDELITY_MIN=0.90`, `REID_MAX=0.01`, `MEMBERSHIP_MAX=0.55`.

### 3.6 Orchestrator Strategy (`orchestrator.py`) — LLM
- **Role:** On failure, diagnose root cause from metrics and produce a concrete, actionable instruction for the Generator.
- **Example output:** "Re-id concentrated on rare ZIP combos; generalize ZIP to 3-digit prefix and add ±5yr age jitter for rare rows; keep common rows sharp to preserve fidelity."
- **Loop control:** `max_rounds=5`; stop on pass or exhaustion.

### 3.7 Report Writer & Publisher (`reporter.py`) — LLM + CODE
- **LLM:** compose human-readable scorecard narrative + reasoning.
- **CODE:** write final synthetic table to BigQuery (`google-cloud-bigquery`); offer CSV export.

---

## 4. Technology Stack

| Layer | Technology |
| :--- | :--- |
| Language Runtime | Python 3.11-slim |
| Web Framework | FastAPI 0.110+ with Uvicorn ASGI |
| Data Validation | Pydantic V2 (structured LLM output via `response_schema`) |
| GCP AI | `google-cloud-aiplatform` (Vertex AI SDK), Gemini 2.x |
| Data Warehouse | `google-cloud-bigquery` (source + sink) |
| Stats / ML | `pandas`, `numpy`, `scipy`, `scikit-learn` |
| Containerization | Docker (multi-stage) |
| Hosting | Cloud Run (2 CPU, 2GiB RAM, autoscale 0-5) |
| CI/CD | Cloud Build + Artifact Registry |
| Front-End | Streamlit (fast) or React (polished) |

---

## 5. Data Contracts (Pydantic V2)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ColumnProfile(BaseModel):
    name: str
    dtype: str  # "numeric" | "categorical"
    stats: Dict[str, Any]  # mean/std/min/max/quantiles or value frequencies
    is_quasi_identifier: bool = False
    is_sensitive: bool = False

class DatasetProfile(BaseModel):
    columns: List[ColumnProfile]
    correlations: Dict[str, float]  # "colA|colB" -> correlation
    row_count: int

class GenerationRequest(BaseModel):
    profile: DatasetProfile
    business_rules: Optional[str] = None
    correction_feedback: Optional[str] = None
    n_rows: int = 500

class UtilityResult(BaseModel):
    fidelity_score: float                    # L1: distribution similarity
    per_column_distance: Dict[str, float]
    correlation_delta: float                 # L2: correlation preservation
    query_equivalence: float                 # L3: analytical SQL match %
    utility_ratio: float                     # L4: TSTR acc_synth / acc_real
    tstr_detail: Dict[str, float] = {}       # {"acc_real": .., "acc_synth": .., "task": ..}

class AttackResult(BaseModel):
    re_id_rate: float
    membership_accuracy: float
    flagged_examples: List[Dict[str, Any]] = []
    attacker_strategy: str

class RoundResult(BaseModel):
    round_index: int
    utility: UtilityResult
    attack: AttackResult
    passed: bool

class ForgeResponse(BaseModel):
    status: str  # "SUCCESS" | "FAILED_TO_CONVERGE"
    rounds: List[RoundResult]
    final_table_id: Optional[str] = None
    scorecard_summary: str
```

---

## 6. API Endpoints

### 6.1 Health Probe
- **`GET /health`** → `{"status":"healthy","service":"synthetic-data-forge","version":"1.0.0"}`

### 6.2 Forge Synthetic Data
- **`POST /api/v1/forge`**
- **Request:**
  ```json
  {
    "source": {"type": "bigquery", "table_id": "hospital.patients"},
    "quasi_identifiers": ["age", "zip", "gender"],
    "sensitive_columns": ["diagnosis"],
    "business_rules": "Preserve age-vs-cost correlation for research.",
    "n_rows": 5000,
    "thresholds": {"fidelity_min": 0.90, "reid_max": 0.01, "membership_max": 0.55}
  }
  ```
- **Response `200 OK`:** a `ForgeResponse` (see contract) with per-round metrics, final BigQuery table id, and scorecard.

### 6.3 Export
- **`GET /api/v1/export/{run_id}?format=csv`** → downloadable synthetic dataset.

---

## 7. Core Algorithms (Reference)

### 7.1 Re-identification (nearest-neighbor)
```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(real_encoded)
distances, _ = nn.kneighbors(synth_encoded)
re_id_rate = (distances < THRESHOLD).mean()
```

### 7.2 Membership inference
```python
# clue = distance from each real record to nearest synthetic row
# label = 1 if record was IN (used to build profile), else 0
clf = LogisticRegression().fit(train_clues, train_labels)
membership_accuracy = clf.score(test_clues, test_labels)  # ~0.5 == safe
```

Encoding: numeric columns scaled; categorical columns one-hot or ordinal encoded; consistent transformer fit on the union of real+synthetic.

### 7.3 Utility proof — TSTR (the headline metric)
```python
# TRTR baseline: train on real, test on held-out real
model_real = RandomForestClassifier().fit(real_train_X, real_train_y)
acc_real = model_real.score(real_test_X, real_test_y)      # e.g., 0.87

# TSTR: train on synthetic, test on the SAME held-out real
model_synth = RandomForestClassifier().fit(synth_X, synth_y)
acc_synth = model_synth.score(real_test_X, real_test_y)    # e.g., 0.83

utility_ratio = acc_synth / acc_real                        # 0.95 -> "95% as useful"
```
The task target column is chosen from the profile (a meaningful categorical/label column). The same held-out REAL test set is used for both, so the comparison is fair.

### 7.4 Query equivalence (demo-friendly)
```python
# Run identical GROUP BY aggregates on real vs synthetic, compare
real_agg  = real_df.groupby(col)[metric].agg(["mean", "count"])
synth_agg = synth_df.groupby(col)[metric].agg(["mean", "count"])
query_equivalence = 1 - (abs(real_agg - synth_agg) / real_agg).mean().mean()
```

---

## 8. Deployment

### 8.1 Dockerfile
```dockerfile
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 2
```

### 8.2 Deploy Script (`deploy.sh`)
```bash
#!/usr/bin/env bash
set -eo pipefail
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="synthetic-data-forge"
IMAGE_URI="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:v1.0.0"

gcloud builds submit --tag ${IMAGE_URI} .
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_URI} --platform managed --region ${REGION} \
  --allow-unauthenticated --set-env-vars GCP_PROJECT=${PROJECT_ID} \
  --memory 2Gi --cpu 2 --min-instances 0 --max-instances 5 --timeout 300s
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format="value(status.url)"
```

---

## 9. Configuration

| Env Var | Purpose | Default |
| :--- | :--- | :--- |
| `GCP_PROJECT` | GCP project id | (required) |
| `VERTEX_LOCATION` | Vertex AI region | `us-central1` |
| `GENERATOR_MODEL` | Generator agent model id | `gemini-2.0-flash` |
| `ATTACKER_MODEL` | Attacker agent model id (tool-calling) | `gemini-2.0-flash` |
| `ORCHESTRATOR_MODEL` | Orchestrator strategy model id | `gemini-2.0-pro` |
| `MAX_ROUNDS` | Max adversarial rounds | `5` |
| `FIDELITY_MIN` | Utility pass threshold | `0.90` |
| `REID_MAX` | Re-id pass threshold | `0.01` |
| `MEMBERSHIP_MAX` | Membership pass threshold | `0.55` |

---

## 10. Project Structure

```
migrator/
├── main.py                 # FastAPI app + routes
├── agents/
│   ├── profiler.py         # CODE: statistical profiling
│   ├── generator.py        # LLM: synthetic row generation
│   ├── statistician.py     # CODE: utility metrics
│   ├── attacker.py         # LLM AGENT: reasons + calls attack tools
│   ├── attack_tools.py     # CODE: re-id, membership inference, outlier linkage
│   └── orchestrator.py     # loop control + LLM strategy + decision gate
├── reporter.py             # LLM narrative + BigQuery publish
├── models.py               # Pydantic V2 contracts
├── config.py               # env config
├── requirements.txt
├── Dockerfile
├── deploy.sh
└── tests/                  # pytest unit tests
```

---

## 11. Testing & Verification

1. **Unit tests (`pytest`):** profiler stats correctness, encoders, metric functions on known fixtures.
2. **Attack sanity tests:** feed an *identical copy* of real data as "synthetic" → re-id rate must be ~100% and membership accuracy high (proves attacks actually detect leakage).
3. **Utility proof test:** compute TSTR on a seeded dataset; assert `utility_ratio` within expected band. Sanity check: a random-noise "synthetic" set must yield a *low* utility_ratio (proves the metric actually discriminates useful vs useless data).
4. **Convergence test:** run full loop on a seeded dataset; assert thresholds met within `MAX_ROUNDS`.
5. **Generality test:** run unchanged on ≥2 datasets (health, finance) and confirm valid output + scorecard.
6. **Local dev:** `uvicorn main:app --reload --port 8080`.

---

## 12. Security & Privacy Notes

- Raw source rows are **never** sent to the LLM — only aggregate statistics.
- Demo uses `--allow-unauthenticated`; production must add IAM / API-key gating (roadmap).
- Service account: least-privilege BigQuery read (source) + write (sink dataset) only.
- Synthetic outputs are validated by the attack suite before publish.
