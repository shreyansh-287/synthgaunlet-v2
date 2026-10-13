"""Feature 10 policy presets and natural-language interpretation."""

import pytest

from agents import policy
from main import PolicyRequest, interpret_policy, policy_presets


def test_presets_cover_balanced_privacy_and_analytics():
    payload = policy_presets()
    presets = {item["id"]: item for item in payload["presets"]}

    assert set(presets) == {"balanced", "privacy_first", "analytics_first"}
    assert (
        presets["privacy_first"]["thresholds"]["reid_max"]
        < presets["balanced"]["thresholds"]["reid_max"]
    )
    assert (
        presets["analytics_first"]["thresholds"]["fidelity_min"]
        > presets["balanced"]["thresholds"]["fidelity_min"]
    )


def test_offline_interpreter_applies_explicit_percentages(monkeypatch):
    monkeypatch.setattr(policy.llm, "is_enabled", lambda: False)

    result = interpret_policy(
        PolicyRequest(
            request=(
                "Prioritize privacy. Keep re-identification below 0.5%, "
                "membership below 55%, and linkage below 8%."
            )
        )
    )

    assert result["source"] == "deterministic"
    assert result["thresholds"]["reid_max"] == pytest.approx(0.005)
    assert result["thresholds"]["membership_max"] == pytest.approx(0.55)
    assert result["thresholds"]["linkage_max"] == pytest.approx(0.08)
    assert "legal compliance" in result["explanation"]


def test_gemini_policy_is_bounded_and_does_not_decide_release(monkeypatch):
    monkeypatch.setattr(policy.llm, "is_enabled", lambda: True)

    def fake_generate(prompt, schema, **kwargs):
        assert "No dataset rows or records are included" in prompt
        assert "re-identification below 0.5%" in prompt
        return schema(
            relevant=True,
            fidelity_min=0.91,
            tstr_min=0.92,
            query_equivalence_min=0.97,
            reid_max=0.004,
            membership_max=0.54,
            attribute_inference_max=0.12,
            linkage_max=0.07,
            near_duplicate_max=0.04,
            max_rounds=5,
            policy_name="Strict technical privacy",
            explanation="Tighter privacy ceilings and high utility floors.",
        )

    monkeypatch.setattr(policy.llm, "generate_structured", fake_generate)
    result = policy.interpret("I need strict privacy and strong model utility.")

    assert result.source == "gemini"
    assert result.thresholds.reid_max == pytest.approx(0.004)
    assert result.policy_name == "Strict technical privacy"
    assert result.explanation == "Tighter privacy ceilings and high utility floors."


def test_irrelevant_or_failed_model_keeps_local_policy(monkeypatch):
    monkeypatch.setattr(policy.llm, "is_enabled", lambda: True)
    request = "Tell me a joke about the weather."

    monkeypatch.setattr(
        policy.llm,
        "generate_structured",
        lambda *args, **kwargs: policy._LLMPolicy(
            relevant=False,
            fidelity_min=0.99,
            tstr_min=0.99,
            query_equivalence_min=0.99,
            reid_max=0.001,
            membership_max=0.51,
            attribute_inference_max=0.01,
            linkage_max=0.01,
            near_duplicate_max=0.01,
            max_rounds=1,
            policy_name="Unused",
            explanation="Unused",
        ),
    )
    irrelevant = policy.interpret(request)
    local = policy._deterministic_interpretation(request)
    assert irrelevant.source == "deterministic"
    assert irrelevant.explanation == local.explanation
    assert irrelevant.thresholds == local.thresholds

    monkeypatch.setattr(policy.llm, "generate_structured", lambda *args, **kwargs: None)
    failed = policy.interpret(
        "Prioritize privacy. Keep re-identification below 0.5%."
    )
    assert failed.source == "deterministic"
    assert failed.thresholds.reid_max == pytest.approx(0.005)
    assert "Interpreted locally" in failed.explanation
