"""Environment-driven configuration for Synthetic Data Forge."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "data" / "output"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Gemini / Vertex ---
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    google_genai_use_vertexai: bool = False
    gcp_project: str = ""
    vertex_location: str = "us-central1"
    google_client_id: str = ""

    generator_model: str = "gemini-3.6-flash"
    attacker_model: str = "gemini-3.6-flash"
    orchestrator_model: str = "gemini-3.6-flash"
    reporter_model: str = "gemini-3.6-flash"
    # Give Gemini time to finish. A local fallback is used only if the call fails.
    llm_timeout_ms: int = 30000
    llm_max_retries: int = 3

    # --- Adversarial loop thresholds ---
    max_rounds: int = 5
    fidelity_min: float = 0.82
    # DCR-excess re-identification bar: fraction of synthetic rows abnormally
    # closer to a rare real individual than a genuine held-out real person.
    tstr_min: float = 0.85
    query_equivalence_min: float = 0.95
    reid_max: float = 0.01
    membership_max: float = 0.60
    default_n_rows: int = 2000
    reid_threshold: float = 0.10

    # --- BigQuery history ---
    bq_history_dataset: str = "synthgauntlet"
    bq_history_table: str = "sidebar_history"
    bq_history_location: str = "US"
    history_limit: int = 25

    # --- Server ---
    port: int = 8080
    public_base_url: str = "http://127.0.0.1:8080"
    cert_signing_key: str = "synthgauntlet-local-cert-key"

    @property
    def gemini_configured(self) -> bool:
        if self.google_genai_use_vertexai:
            return bool(self.gcp_project)
        return bool(self.gemini_api_key)

    @property
    def effective_provider(self) -> str:
        """Resolve the Gemini backend, or deterministic mode when unavailable."""
        provider = (self.llm_provider or "gemini").lower()
        if provider not in {"gemini", "auto"}:
            return "none"
        return "gemini" if self.gemini_configured else "none"

    @property
    def llm_configured(self) -> bool:
        return self.effective_provider != "none"


def _apply_constants_defaults(settings: Settings) -> Settings:
    """Environment variables win; otherwise fall back to values in constants.py."""
    try:
        import constants
    except Exception:  # pragma: no cover
        return settings

    if getattr(constants, "GEMINI_API_KEY", "") and not settings.gemini_api_key:
        settings.gemini_api_key = constants.GEMINI_API_KEY
    return settings


@lru_cache()
def get_settings() -> Settings:
    return _apply_constants_defaults(Settings())
