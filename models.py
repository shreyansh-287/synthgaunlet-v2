"""Pydantic V2 data contracts for Synthetic Data Forge.

`Code measures. The LLM decides and creates.` These models are the typed
boundary between the deterministic verifiers and the LLM agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
class ColumnProfile(BaseModel):
    name: str
    dtype: str  # "numeric" | "categorical"
    stats: Dict[str, Any]  # mean/std/min/max/quantiles OR value frequencies
    missing_rate: float = Field(0.0, ge=0.0, le=1.0)
    is_quasi_identifier: bool = False
    is_sensitive: bool = False


class SemanticProfile(BaseModel):
    """Semantic analysis produced by LLM Profile Analyst."""

    functional_dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Inferred child->parent column dependencies (e.g. {'JobRole': 'Department'})",
    )
    high_risk_qi_combos: List[List[str]] = Field(
        default_factory=list,
        description="Quasi-identifier combinations prone to high re-identification uniqueness.",
    )
    business_constraints: List[str] = Field(
        default_factory=list,
        description="Natural business constraints or domain invariants.",
    )
    conditional_generation_order: List[str] = Field(
        default_factory=list,
        description="Topological column order for conditional / hierarchical synthesis.",
    )
    analyst_notes: str = ""


class DatasetProfile(BaseModel):
    columns: List[ColumnProfile]
    correlations: Dict[str, float] = Field(default_factory=dict)  # "a|b" -> corr
    missingness: Dict[str, float] = Field(default_factory=dict)
    rare_combinations: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int
    label_column: Optional[str] = None
    semantic_profile: Optional[SemanticProfile] = None


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class SynthesisConfig(BaseModel):
    """The Generator agent's privacy/utility knobs. The LLM picks these values;
    deterministic code samples rows accordingly."""

    plan_name: str = Field("balanced", description="Role/tag of candidate: high_utility, balanced, high_privacy")
    recombination: float = Field(
        0.1,
        description=(
            "Core utility<->privacy dial. 0 = synthetic rows are near-copies of"
            " real people (max utility, max leakage). 1 = every value is drawn"
            " independently from the label-conditional marginals (max privacy)."
        ),
    )
    numeric_noise: float = Field(
        0.05,
        description="Gaussian jitter added to numeric columns, as a fraction of std.",
    )
    categorical_shuffle: float = Field(
        0.05,
        description="Probability of resampling a categorical value from its marginal.",
    )
    generalize_columns: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-column bucket size for numeric generalization (0 = off).",
    )
    rare_category_floor: float = Field(
        0.01,
        description="Categories rarer than this frequency get blurred/merged.",
    )
    group_dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping child->parent column for hierarchical sampling.",
    )
    copula_rank_preservation: bool = Field(
        True,
        description="Whether to preserve continuous rank order to maintain covariance.",
    )
    min_max_bounds: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Per-column [min, max] domain clamping bounds.",
    )
    rationale: str = ""


class CandidatePlanBatch(BaseModel):
    """Batch of candidate generation plans proposed by Generator LLM."""

    candidates: List[SynthesisConfig] = Field(default_factory=list)
    selection_strategy: str = ""


class GenerationRequest(BaseModel):
    profile: DatasetProfile
    business_rules: Optional[str] = None
    correction_feedback: Optional[str] = None
    n_rows: int = 2000


# --------------------------------------------------------------------------- #
# Utility (Statistician)
# --------------------------------------------------------------------------- #
class BrokenRelationship(BaseModel):
    col_a: str
    col_b: str
    real_corr: float
    synth_corr: float
    delta: float


class DiagnosticReport(BaseModel):
    weakest_columns: List[str] = Field(default_factory=list)
    broken_relationships: List[BrokenRelationship] = Field(default_factory=list)
    utility_choke_points: List[str] = Field(default_factory=list)
    per_column_ks: Dict[str, float] = Field(default_factory=dict)


class AuditMetrics(BaseModel):
    fidelity_score: float                 # L1: distribution similarity (0..1)
    per_column_distance: Dict[str, float] = Field(default_factory=dict)
    correlation_delta: float = 0.0        # L2: correlation preservation
    query_equivalence: float = 0.0        # L3: analytical SQL match %
    utility_ratio: float = 0.0            # L4: TSTR acc_synth / acc_real
    tstr_detail: Dict[str, Any] = Field(default_factory=dict)
    diagnostic_report: Optional[DiagnosticReport] = None


# --------------------------------------------------------------------------- #
# Attack (Attacker)
# --------------------------------------------------------------------------- #
class PrivacyAttackResults(BaseModel):
    re_id_rate: float
    membership_accuracy: float
    outlier_linkage_rate: float = 0.0
    flagged_examples: List[Dict[str, Any]] = Field(default_factory=list)
    attacker_strategy: str = ""
    tools_used: List[str] = Field(default_factory=list)
    attacked_columns: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Candidate Evaluation & Loop
# --------------------------------------------------------------------------- #
class CandidateEvaluation(BaseModel):
    candidate_index: int
    plan: SynthesisConfig
    utility: AuditMetrics
    attack: PrivacyAttackResults
    passed: bool
    composite_score: float = 0.0


class RoundResult(BaseModel):
    round_index: int
    utility: AuditMetrics
    attack: PrivacyAttackResults
    passed: bool
    plan: SynthesisConfig
    orchestrator_feedback: str = ""
    evaluated_candidates: List[CandidateEvaluation] = Field(default_factory=list)
    winning_candidate_index: int = 0


class RunBlackboard(BaseModel):
    """Full cross-round memory blackboard persistent across all iterations."""

    run_id: str
    semantic_profile: Optional[SemanticProfile] = None
    round_history: List[RoundResult] = Field(default_factory=list)
    weakest_columns_history: List[str] = Field(default_factory=list)
    broken_relationships_history: List[BrokenRelationship] = Field(default_factory=list)
    active_attack_vectors: List[Dict[str, Any]] = Field(default_factory=list)
    orchestrator_feedback_history: List[str] = Field(default_factory=list)


class Thresholds(BaseModel):
    max_rounds: int = Field(5, ge=1, le=5)
    fidelity_min: float = Field(0.82, ge=0.0, le=1.0)
    tstr_min: float = Field(0.85, ge=0.0, le=1.0)
    query_equivalence_min: float = Field(0.95, ge=0.0, le=1.0)
    reid_max: float = Field(0.01, ge=0.0, le=1.0)
    membership_max: float = Field(0.60, ge=0.0, le=1.0)


class DualScorecard(BaseModel):
    status: str
    utility: AuditMetrics
    privacy: PrivacyAttackResults
    thresholds: Thresholds
    passed: bool
    rounds: int = Field(ge=1)


GenerationPlan = SynthesisConfig
UtilityResult = AuditMetrics
AttackResult = PrivacyAttackResults


class SourceSpec(BaseModel):
    type: str = "csv"                 # "csv" | "upload" | "bigquery"
    dataset_key: Optional[str] = None  # key into constants.DATASETS (csv mode)
    upload_id: Optional[str] = None    # token for a user-uploaded CSV
    table_id: Optional[str] = None     # BigQuery table (later)


class ForgeRequest(BaseModel):
    source: SourceSpec
    quasi_identifiers: Optional[List[str]] = None
    sensitive_columns: Optional[List[str]] = None
    label_column: Optional[str] = None
    business_rules: Optional[str] = None
    n_rows: Optional[int] = Field(None, ge=1, le=50_000)
    destination_table_id: Optional[str] = None
    thresholds: Optional[Thresholds] = None


class ForgeResponse(BaseModel):
    run_id: str
    status: str  # "SUCCESS" | "FAILED_TO_CONVERGE"
    dataset_key: Optional[str] = None
    domain: Optional[str] = None
    rounds: List[RoundResult] = Field(default_factory=list)
    final_table_id: Optional[str] = None
    export_path: Optional[str] = None
    scorecard_summary: str = ""
    scorecard: Optional[DualScorecard] = None
    thresholds: Thresholds = Field(default_factory=Thresholds)
    llm_enabled: bool = False
    difficulty: str = ""          # "easy" | "moderate" | "hard"
    difficulty_detail: str = ""   # plain-English why


class RunStartResponse(BaseModel):
    run_id: str
    status: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: List[dict] = Field(default_factory=list)
    response: Optional[ForgeResponse] = None
    error: Optional[str] = None
