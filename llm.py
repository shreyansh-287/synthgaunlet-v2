"""Gemini wrapper used only for agent strategy and narrative generation."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from config import get_settings

logger = logging.getLogger("forge.llm")

T = TypeVar("T", bound=BaseModel)

_gemini_client = None
_init_attempted = False
_llm_failed = False


def _get_gemini_client():
    global _gemini_client, _init_attempted
    if _init_attempted:
        return _gemini_client
    _init_attempted = True
    settings = get_settings()
    if not settings.gemini_configured:
        return None
    try:
        from google import genai
        from google.genai import types

        fast_http = types.HttpOptions(timeout=60000)

        if settings.google_genai_use_vertexai:
            _gemini_client = genai.Client(
                vertexai=True,
                project=settings.gcp_project,
                location=settings.vertex_location,
                http_options=fast_http,
            )
            logger.info(
                "Vertex AI Gemini client initialised (project=%s, location=%s).",
                settings.gcp_project,
                settings.vertex_location,
            )
        else:
            _gemini_client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=fast_http,
            )
            logger.info("Gemini Developer API client initialised.")
    except Exception as exc:  # noqa: BLE001
        if settings.google_genai_use_vertexai:
            logger.warning(
                "Vertex AI client init failed (%s). Ensure ADC is configured via 'gcloud auth application-default'. "
                "Public REST fallback is disabled in Vertex mode.",
                exc,
            )
        else:
            logger.info("GenAI SDK not available locally (%s); using REST API fallback.", exc)
    return _gemini_client


def reset_llm_state() -> None:
    global _llm_failed, _init_attempted, _RUN_LOG
    _llm_failed = False
    _init_attempted = False
    _RUN_LOG = []


def get_run_log() -> list[Dict[str, Any]]:
    """Advisory calls made during the current forge run."""
    return list(_RUN_LOG)


def record(role: str, purpose: str, model: str, used_model: bool) -> None:
    """Remember an advisory call that already happened, such as policy interpretation."""
    if not role:
        return
    _RUN_LOG.append({
        "role": role,
        "purpose": purpose,
        "model": model,
        "used_model": bool(used_model),
    })


def _record(role: str, purpose: str, model: str, used_model: bool) -> None:
    record(role, purpose, model, used_model)


def is_enabled() -> bool:
    if _llm_failed:
        return False
    settings = get_settings()
    return bool(settings.gemini_configured)


def provider_name() -> str:
    return "gemini" if is_enabled() else "none"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


_TOTAL_INPUT_TOKENS = 0
_TOTAL_OUTPUT_TOKENS = 0
_TOTAL_LLM_CALLS = 0
_RUN_LOG: list[Dict[str, Any]] = []


def get_token_usage() -> Dict[str, int]:
    """Return cumulative token usage stats across the application lifetime."""
    return {
        "total_calls": _TOTAL_LLM_CALLS,
        "input_tokens": _TOTAL_INPUT_TOKENS,
        "output_tokens": _TOTAL_OUTPUT_TOKENS,
        "total_tokens": _TOTAL_INPUT_TOKENS + _TOTAL_OUTPUT_TOKENS,
    }


def _log_tokens(model: str, prompt_tokens: int, output_tokens: int) -> None:
    global _TOTAL_INPUT_TOKENS, _TOTAL_OUTPUT_TOKENS, _TOTAL_LLM_CALLS
    _TOTAL_INPUT_TOKENS += prompt_tokens
    _TOTAL_OUTPUT_TOKENS += output_tokens
    _TOTAL_LLM_CALLS += 1
    call_total = prompt_tokens + output_tokens
    cum_total = _TOTAL_INPUT_TOKENS + _TOTAL_OUTPUT_TOKENS
    print(
        f"[Gemini Token Usage] Call #{_TOTAL_LLM_CALLS} ({model}) | "
        f"Prompt: {prompt_tokens:,} | Response: {output_tokens:,} | "
        f"Call Total: {call_total:,} | Cumulative Total: {cum_total:,} tokens"
    )
    logger.info(
        "Gemini Token Usage - Call #%d (%s): prompt=%d, response=%d, total=%d (cumulative=%d)",
        _TOTAL_LLM_CALLS, model, prompt_tokens, output_tokens, call_total, cum_total,
    )


def _call_with_backoff(callable_fn, max_retries: int = 3):
    """Execute LLM call with exponential backoff (1s, 2s, 4s) on transient errors."""
    for attempt in range(max_retries):
        try:
            return callable_fn()
        except Exception as exc:
            exc_str = str(exc)
            is_transient = any(
                code in exc_str
                for code in [
                    "429", "500", "503", "504",
                    "ResourceExhausted", "InternalServerError",
                    "ServiceUnavailable", "DeadlineExceeded", "timeout",
                ]
            )
            if is_transient and attempt < max_retries - 1:
                wait_s = 1.0 * (2 ** attempt)
                logger.warning(
                    "Transient Vertex AI error (%s); retrying in %.1fs (attempt %d/%d)...",
                    exc, wait_s, attempt + 1, max_retries,
                )
                time.sleep(wait_s)
                continue
            raise exc


def _rest_generate(
    prompt: str,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> Optional[str]:
    settings = get_settings()
    if settings.google_genai_use_vertexai:
        # Public REST fallback is strictly prohibited when Vertex AI mode is active
        return None

    api_key = settings.gemini_api_key
    if not api_key:
        return None

    models_to_try = []
    for m in [model, "gemini-flash-lite-latest"]:
        if m and m not in models_to_try:
            models_to_try.append(m)

    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload_dict: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if json_mode:
            payload_dict["generationConfig"]["responseMimeType"] = "application/json"
        if system_instruction:
            payload_dict["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload_dict).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                usage = res_data.get("usageMetadata", {})
                p_tok = usage.get("promptTokenCount", len(prompt) // 4)
                o_tok = usage.get("candidatesTokenCount", 100)
                _log_tokens(m, p_tok, o_tok)
                return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini REST model %s attempt failed: %s", m, exc)
            continue

    return None


def generate_text(
    prompt: str,
    *,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    role: str = "",
    purpose: str = "",
) -> Optional[str]:
    text = _generate_text(prompt, model=model, system_instruction=system_instruction, temperature=temperature)
    _record(role, purpose, model, text is not None)
    return text


def _generate_text(
    prompt: str,
    *,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
) -> Optional[str]:
    settings = get_settings()
    client = _get_gemini_client()
    if client is not None:
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
            )
            response = _call_with_backoff(
                lambda: client.models.generate_content(model=model, contents=prompt, config=config)
            )
            usage = getattr(response, "usage_metadata", None)
            if usage:
                _log_tokens(model, getattr(usage, "prompt_token_count", 0), getattr(usage, "candidates_token_count", 0))
            return (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini SDK generate_text failed (%s): %s", model, exc)
            if settings.google_genai_use_vertexai:
                return None

    if settings.google_genai_use_vertexai:
        return None

    return _rest_generate(
        prompt, model=model, system_instruction=system_instruction, temperature=temperature
    )


def generate_json(
    prompt: str,
    *,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.4,
    role: str = "",
    purpose: str = "",
) -> Optional[Dict[str, Any]]:
    data = _generate_json(prompt, model=model, system_instruction=system_instruction, temperature=temperature)
    _record(role, purpose, model, data is not None)
    return data


def _generate_json(
    prompt: str,
    *,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.4,
) -> Optional[Dict[str, Any]]:
    settings = get_settings()
    client = _get_gemini_client()
    if client is not None:
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                response_mime_type="application/json",
            )
            response = _call_with_backoff(
                lambda: client.models.generate_content(model=model, contents=prompt, config=config)
            )
            usage = getattr(response, "usage_metadata", None)
            if usage:
                _log_tokens(model, getattr(usage, "prompt_token_count", 0), getattr(usage, "candidates_token_count", 0))
            return _extract_json(response.text or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini SDK generate_json failed (%s): %s", model, exc)
            if settings.google_genai_use_vertexai:
                return None

    if settings.google_genai_use_vertexai:
        return None

    text = _rest_generate(
        prompt, model=model, system_instruction=system_instruction, temperature=temperature, json_mode=True
    )
    return _extract_json(text or "")


def generate_structured(
    prompt: str,
    response_schema: Type[T],
    *,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.4,
    role: str = "",
    purpose: str = "",
) -> Optional[T]:
    """Strict structured output parsing backed by Pydantic response schema."""
    parsed = _generate_structured(
        prompt, response_schema, model=model, system_instruction=system_instruction, temperature=temperature,
    )
    _record(role, purpose, model, parsed is not None)
    return parsed


def _generate_structured(
    prompt: str,
    response_schema: Type[T],
    *,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.4,
) -> Optional[T]:
    settings = get_settings()
    client = _get_gemini_client()
    if client is not None:
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
            )
            response = _call_with_backoff(
                lambda: client.models.generate_content(model=model, contents=prompt, config=config)
            )
            usage = getattr(response, "usage_metadata", None)
            if usage:
                _log_tokens(model, getattr(usage, "prompt_token_count", 0), getattr(usage, "candidates_token_count", 0))
            raw_text = (response.text or "").strip()
            raw_dict = _extract_json(raw_text)
            if raw_dict is not None:
                try:
                    return response_schema.model_validate(raw_dict)
                except Exception:
                    return response_schema(**raw_dict)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini SDK generate_structured failed (%s): %s", model, exc)
            if settings.google_genai_use_vertexai:
                return None

    if settings.google_genai_use_vertexai:
        return None

    text = _rest_generate(
        prompt, model=model, system_instruction=system_instruction, temperature=temperature, json_mode=True
    )
    raw_dict = _extract_json(text or "")
    if raw_dict is not None:
        try:
            return response_schema.model_validate(raw_dict)
        except Exception:
            try:
                return response_schema(**raw_dict)
            except Exception as exc:
                logger.warning("Structured parsing failed on fallback response: %s", exc)
    return None


def advise(role: str, purpose: str, prompt: str, *, model: str, temperature: float = 0.3) -> str:
    """Ask Gemini for a short explanation. An empty string means the deterministic path stands."""
    text = generate_text(
        prompt,
        model=model,
        role=role,
        purpose=purpose,
        temperature=temperature,
        system_instruction=(
            "You advise a synthetic-data release system. Use only the numbers and names in the prompt. "
            "Do not invent rows, people, or approval. Reply in two sentences at most."
        ),
    )
    return (text or "").strip()
