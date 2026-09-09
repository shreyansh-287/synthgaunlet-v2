# Product Requirements Document (PRD): Synthetic Data Forge

**Project Name:** Synthetic Data Forge — Adversarial Multi-Agent Synthetic Data Engine
**Cohort Focus:** Multi-Agent AI Systems on Google Cloud Platform (GCP)
**Document Owner:** [You]
**Status:** Draft v1.0
**Target Audience:** Google AI Cohort Panelists, Data Engineers, Privacy/Compliance Officers, ML Teams

---

## 1. Executive Summary

Enterprises sit on vast amounts of sensitive data (patient records, financial transactions, HR data) that could power valuable ML models and analytics — but **privacy law (GDPR, HIPAA) prevents them from sharing or freely using it.** The result is a massive innovation bottleneck.

**Synthetic Data Forge** is an autonomous, adversarial multi-agent system on GCP that generates synthetic tabular datasets which are **statistically faithful** to the original data yet **provably resistant to privacy attacks**. A Generator agent creates fake data; an adaptive Attacker agent tries to break its privacy; an Orchestrator agent referees the two through iterative rounds until the data is simultaneously **useful and safe** — verified by deterministic statistical tests, not by the AI's own opinion.

---

## 2. Problem Statement

| # | Problem | Impact |
| :-- | :--- | :--- |
| P1 | Sensitive data cannot be shared with external partners or moved to less-secure dev/test environments. | Projects stall for weeks awaiting legal approval, or never happen. |
| P2 | Naive anonymization (dropping names) still leaks identity via quasi-identifiers (age + ZIP + gender). | Re-identification attacks expose real individuals. |
| P3 | Existing synthetic-data tools rarely **prove** the output is private. | Teams can't trust or defend the safety of shared data. |
| P4 | Utility and privacy are in direct tension; tuning the tradeoff is manual and expert-heavy. | Data is either too risky to share or too degraded to use. |

---

## 3. Product Vision

> A schema-agnostic engine that takes **any tabular dataset** and returns a synthetic version that is realistic enough to train models on, and carries a **Privacy Scorecard proving** that even an adaptive AI attacker cannot re-identify individuals or infer dataset membership.

---

## 4. Goals & Non-Goals

### 4.1 Goals (MVP)
- **G1:** Accept any tabular dataset (CSV upload or BigQuery table) with arbitrary schema — no per-dataset code changes.
- **G2:** Generate synthetic data via an LLM Generator agent that respects distributions and column correlations.
- **G3:** Run an adaptive adversarial Attacker agent performing two attacks: re-identification and membership inference.
- **G4:** Iterate a self-improving loop (Generator ↔ Attacker, refereed by Orchestrator) until utility and privacy thresholds are met.
- **G5:** Produce a **Privacy Scorecard** with deterministic metrics (fidelity %, re-identification %, membership-inference accuracy %).
- **G6:** Publish the safe synthetic dataset back to BigQuery and/or offer download.
- **G7:** Demonstrate generality by running the same pipeline on 2-3 datasets across domains.

### 4.2 Non-Goals (explicitly out of scope for cohort)
- **NG1:** Free-text, image, audio, or time-series data (tabular only).
- **NG2:** Millions-of-rows batch scale (representative sampling of thousands of rows is sufficient).
- **NG3:** Formal differential-privacy mathematical guarantees (we provide empirical attack-based evidence).
- **NG4:** Production multi-tenant auth/billing (demo-grade access only).

---

## 5. User Personas & Use Cases

### 5.1 The Compliance-Bound Data Owner (e.g., Hospital DBA)
- **Goal:** Let an external ML partner work with realistic data without exposing patients.
- **Pain:** Legal cannot approve sharing real records.
- **Solution:** Upload the table → receive safe synthetic data + a scorecard to hand to legal.

### 5.2 The ML Engineer / External Partner
- **Goal:** Build and test models quickly without waiting for data-access approval.
- **Pain:** Weeks of bureaucracy before touching data.
- **Solution:** Develop on synthetic data instantly; the final model runs on real data internally.

### 5.3 The Privacy / Compliance Officer
- **Goal:** Prove to auditors that shared data cannot re-identify individuals.
- **Pain:** No quantitative evidence of safety.
- **Solution:** The Privacy Scorecard provides defensible, measurable attack-resistance numbers.

---

## 6. Key Features & Requirements

| ID | Feature | Requirement | Priority |
| :-- | :--- | :--- | :--- |
| F1 | **Dataset Ingestion** | Accept CSV upload or BigQuery table reference; auto-detect schema and column types. | Must |
| F2 | **Statistical Profiler** | Deterministically compute distributions, ranges, and correlations (no raw data sent to LLM). | Must |
| F3 | **Generator Agent** | LLM produces synthetic rows matching the profile and business rules. | Must |
| F4 | **Utility Evaluation** | Deterministic fidelity score (distribution distance + correlation delta + optional TSTR). | Must |
| F5 | **Adaptive Attacker Agent** | Runs re-identification + membership-inference attacks; adapts strategy each round. | Must |
| F6 | **Orchestrator Loop** | Reads scores, decides fix strategy, feeds back to Generator; max N rounds. | Must |
| F7 | **Privacy Scorecard** | Human-readable report with all metrics, verdict, and reasoning. | Must |
| F8 | **Quasi-Identifier Detection** | User-tagged (MVP) or LLM auto-detected quasi-identifiers. | Should |
| F9 | **Publish / Export** | Write safe synthetic table to BigQuery; offer CSV download. | Must |
| F10 | **Multi-Dataset Demo** | Same engine runs on ≥2 domains (health, finance, HR). | Should |
| F11 | **Live Progress Log** | Stream round-by-round metrics to the UI. | Should |

---

## 7. Success Metrics

| Metric | Target |
| :--- | :--- |
| **Fidelity (distribution similarity)** | ≥ 90% similarity to source |
| **TSTR utility ratio (proven usefulness)** | ≥ 90% (model trained on synthetic works ≥90% as well on real data) |
| **Query equivalence** | ≥ 95% match on analytical aggregates |
| **Re-identification risk** | ≤ 1% of synthetic rows are near-copies |
| **Membership-inference accuracy** | ≤ 55% (≈ coin flip = safe) |
| **Convergence** | Thresholds met within ≤ 5 rounds |
| **Generality** | Runs unmodified on ≥ 2 distinct datasets |
| **Latency** | End-to-end run on a demo dataset < 2 minutes |

---

## 8. User Flow

```
1. User uploads a CSV or selects a BigQuery table.
2. User optionally tags quasi-identifiers and sets a business goal.
3. User clicks "Forge Synthetic Data".
4. UI streams the adversarial rounds live (utility ↑, leakage ↓).
5. On success: user views the Dual Scorecard (Utility proof + Privacy proof).
6. User publishes the safe dataset to BigQuery or downloads CSV.
```

---

## 9. Differentiation (Why We Win)

| Competitor Approach | Our Advantage |
| :--- | :--- |
| Generic "chatbot with agents" demos | Genuine **adversarial** multi-agent duel with adaptation |
| Synthetic-data tools without proof | A **Dual Scorecard**: proven utility (TSTR) *and* proven privacy (real attacks) |
| LLM-only pipelines (untrustworthy numbers) | **Deterministic verifiers** compute every metric — AI can't grade itself |
| Single hard-coded dataset demos | **Schema-agnostic**, proven live on multiple domains |

---

## 10. Demo Script (5-Minute Pitch)

1. **Hook (0:00-1:00):** "This hospital's data could save lives, but privacy law keeps it locked. We unlock it — provably safely."
2. **Architecture (1:00-2:00):** Introduce the adversarial agents: Generator vs adaptive Attacker, refereed by the Orchestrator.
3. **Live Demo (2:00-3:45):** Upload a dataset → watch the rounds converge (leakage 12% → 0.8%) → show histograms (real vs synthetic overlap) → reveal the Scorecard.
4. **Generality (3:45-4:15):** Run the *same* engine on a finance dataset to prove it's a general engine.
5. **Wrap + Q&A (4:15-5:00):** "Realistic, provably private, GCP-native, in seconds."

---

## 11. Roadmap

- **Milestone 1 (Cohort Demo):** Tabular engine, adversarial loop, scorecard, BigQuery publish, multi-dataset demo.
- **Milestone 2:** LLM auto-detection of quasi-identifiers; larger-scale hybrid generation (LLM strategy + code sampling).
- **Milestone 3:** Time-series support; formal differential-privacy option; enterprise auth & audit logging.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| Judge asks "is the attack real?" | Use concrete, named methods (nearest-neighbor re-id, membership-inference classifier). |
| "Is it really multi-agent?" | Make the Attacker LLM-driven and adaptive; Orchestrator reasons about strategy. |
| LLM generation doesn't scale | Scope to representative samples (thousands of rows); frame scale as roadmap. |
| Utility vs privacy has no sweet spot for some data | Report honestly; the scorecard shows the achieved tradeoff. |
