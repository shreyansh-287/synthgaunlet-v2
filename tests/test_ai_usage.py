"""Gemini advisory calls are recorded even when the model is not configured."""

from config import get_settings
import llm


def test_unconfigured_advice_is_logged_as_a_fallback(monkeypatch):
    settings = get_settings().model_copy(update={
        "gemini_api_key": "",
        "google_genai_use_vertexai": False,
    })
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    llm.reset_llm_state()

    text = llm.advise(
        "profiler",
        "Explain the aggregate profile.",
        "Column summaries only. No rows.",
        model="gemini-test",
    )

    assert text == ""
    assert llm.get_run_log() == [{
        "role": "profiler",
        "purpose": "Explain the aggregate profile.",
        "model": "gemini-test",
        "used_model": False,
    }]


def test_a_model_reply_is_marked_as_used(monkeypatch):
    llm.reset_llm_state()
    monkeypatch.setattr(llm, "_generate_text", lambda *args, **kwargs: "The budget was split across the released queries.")

    text = llm.advise(
        "generator",
        "Explain how the Laplace epsilon budget was split.",
        "Epsilon spent 1.0 of 1.0.",
        model="gemini-test",
    )

    assert "budget" in text
    assert llm.get_run_log()[0]["used_model"] is True
