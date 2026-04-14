from __future__ import annotations

import os

from research_report_flow.config import Settings, get_settings


def test_constructor_allows_field_names_over_alias_env() -> None:
    value = Settings(use_live_crews=False, llm_model=None)
    assert value.use_live_crews is False
    assert value.llm_model is None


def test_free_api_mode_applies_conservative_limits(monkeypatch) -> None:
    monkeypatch.setenv("FREE_API_MODE", "true")
    monkeypatch.setenv("USE_LIVE_CREWS", "true")
    monkeypatch.setenv("RESEARCH_REPORT_MODEL", "openrouter/openai/gpt-oss-20b")
    monkeypatch.setenv("RESEARCH_REPORT_OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("SEARCH_RESULT_LIMIT", "10")
    monkeypatch.setenv("SCHOLAR_RESULT_LIMIT", "10")
    monkeypatch.setenv("MAX_AGENDA_ITEMS", "10")
    monkeypatch.setenv("DUAL_RESEARCH_PARALLEL", "true")
    monkeypatch.setenv("API_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("API_RETRY_BACKOFF_SECONDS", "0.1")
    monkeypatch.setenv("LLM_MIN_INTERVAL_SECONDS", "1")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.free_api_mode is True
    assert settings.api_retry_attempts >= 8
    assert settings.api_retry_backoff_seconds >= 2.0
    assert settings.llm_min_interval_seconds >= 13.0
    assert settings.allow_live_fallback is False
    assert settings.search_result_limit <= 3
    assert settings.scholar_result_limit <= 3
    assert settings.max_agenda_items == 2
    assert settings.dual_research_parallel is False

    get_settings.cache_clear()


def test_local_ollama_without_api_key_still_enables_live_crews() -> None:
    settings = Settings(
        USE_LIVE_CREWS=True,
        RESEARCH_REPORT_MODEL="ollama/gemma4:31b-cloud",
        RESEARCH_REPORT_OPENROUTER_BASE_URL="http://localhost:11434/v1",
    )

    assert settings.should_use_live_crews() is True


def test_get_settings_exports_openai_compatible_base_urls_for_local_ollama(monkeypatch) -> None:
    monkeypatch.setenv("USE_LIVE_CREWS", "true")
    monkeypatch.setenv("RESEARCH_REPORT_MODEL", "ollama/gemma4:31b-cloud")
    monkeypatch.setenv("RESEARCH_REPORT_OPENROUTER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("RESEARCH_REPORT_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.should_use_live_crews() is True
    assert os.getenv("OPENAI_BASE_URL") == "http://localhost:11434/v1"
    assert os.getenv("OPENAI_API_BASE") == "http://localhost:11434/v1"

    get_settings.cache_clear()
