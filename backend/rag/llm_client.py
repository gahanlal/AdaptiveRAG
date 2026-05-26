"""Centralized LLM client factory — provider-agnostic.

Supported providers (set LLM_PROVIDER in .env):

  ollama  — local or self-hosted server, no API key needed
            OLLAMA_BASE_URL  (default: http://localhost:11434/v1)
            OLLAMA_MODEL     (default: llama3.2)

  openai  — OpenAI cloud
            OPENAI_API_KEY   (required)
            OPENAI_MODEL     (default: gpt-4o-mini)

  groq    — Groq cloud, free tier, very fast Llama/Mixtral
            GROQ_API_KEY     (required)
            GROQ_MODEL       (default: llama3-8b-8192)

  custom  — any OpenAI-compatible endpoint (Together.ai, vLLM, Anyscale…)
            CUSTOM_BASE_URL  (required)
            CUSTOM_API_KEY   (required)
            CUSTOM_MODEL     (required)

  groq-fallback — try Groq first; if it fails fall back to OpenAI automatically
                   set LLM_PROVIDER=groq-fallback

Usage:
    from backend.rag.llm_client import get_client
    client, model = get_client()
    client.chat.completions.create(model=model, messages=[...])

    # Fallback-aware call:
    from backend.rag.llm_client import chat_with_fallback
    response = chat_with_fallback(messages=[...])
"""
from __future__ import annotations

import logging
import os
from openai import OpenAI

log = logging.getLogger(__name__)

_client: OpenAI | None = None
_current_provider: str | None = None  # detect .env changes between calls

# Cheapest/fastest Groq model (free tier)
_GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"


def _make_groq_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)


def _make_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def chat_with_fallback(messages: list, **kwargs) -> object:
    """Call Groq first; transparently fall back to OpenAI on any error."""
    groq_model = os.environ.get("GROQ_MODEL", _GROQ_DEFAULT_MODEL)
    try:
        client = _make_groq_client()
        return client.chat.completions.create(model=groq_model, messages=messages, **kwargs)
    except Exception as groq_err:
        log.warning("Groq call failed (%s), falling back to OpenAI", groq_err)
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        client = _make_openai_client()
        return client.chat.completions.create(model=openai_model, messages=messages, **kwargs)


def get_client() -> tuple[OpenAI, str]:
    """Return (OpenAI-compatible client, model name) for the configured provider."""
    global _client, _current_provider

    provider = os.environ.get("LLM_PROVIDER", "groq-fallback").lower().strip()

    # Rebuild if provider changed (e.g. .env hot-reloaded)
    if provider != _current_provider:
        _client = None
        _current_provider = provider

    if _client is not None:
        return _client, _model_for(provider)

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _client = OpenAI(base_url=base_url, api_key="ollama")

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env"
            )
        _client = OpenAI(api_key=api_key)

    elif provider == "groq":
        _client = _make_groq_client()

    elif provider == "groq-fallback":
        # Returns a Groq client; actual fallback happens in chat_with_fallback()
        # get_client() returns the primary (Groq) client; callers that want fallback
        # should use chat_with_fallback() directly.
        try:
            _client = _make_groq_client()
            _current_provider = "groq-fallback"
        except EnvironmentError:
            log.warning("GROQ_API_KEY missing, using OpenAI directly")
            _client = _make_openai_client()

    elif provider == "custom":
        base_url = os.environ.get("CUSTOM_BASE_URL", "")
        api_key = os.environ.get("CUSTOM_API_KEY", "")
        if not base_url:
            raise EnvironmentError(
                "LLM_PROVIDER=custom but CUSTOM_BASE_URL is not set in .env"
            )
        _client = OpenAI(base_url=base_url, api_key=api_key or "none")

    else:
        raise EnvironmentError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            "Choose one of: ollama | openai | groq | groq-fallback | custom"
        )

    return _client, _model_for(provider)


def _model_for(provider: str) -> str:
    defaults = {
        "ollama":  ("OLLAMA_MODEL",  "llama3.2"),
        "openai":  ("OPENAI_MODEL",  "gpt-4o-mini"),
        "groq":         ("GROQ_MODEL", _GROQ_DEFAULT_MODEL),
        "groq-fallback": ("GROQ_MODEL", _GROQ_DEFAULT_MODEL),
        "custom":  ("CUSTOM_MODEL",  ""),
    }
    env_var, default = defaults.get(provider, ("OLLAMA_MODEL", "llama3.2"))
    return os.environ.get(env_var, default)


def reset() -> None:
    """Reset cached singleton — used in tests."""
    global _client, _current_provider
    _client = None
    _current_provider = None
