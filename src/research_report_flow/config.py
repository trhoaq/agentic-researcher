from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def configure_runtime_environment(base_dir: Path | None = None) -> Path:
    root = (base_dir or Path.cwd()).resolve()
    local_app_data = root / ".localapp"
    crewai_storage = root / ".crewai"
    tmp_dir = root / ".tmp"

    local_app_data.mkdir(parents=True, exist_ok=True)
    crewai_storage.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LOCALAPPDATA"] = str(local_app_data)
    os.environ["CREWAI_STORAGE_DIR"] = str(crewai_storage)
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    return root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        validate_by_name=True,
    )

    project_root: Path = Field(default_factory=lambda: configure_runtime_environment())
    output_root: Path = Field(default_factory=lambda: configure_runtime_environment() / "output")
    llm_model: str | None = Field(default=None, validation_alias="RESEARCH_REPORT_MODEL")
    research_report_gemini_api_key: str | None = Field(default=None, validation_alias="RESEARCH_REPORT_GEMINI_API_KEY")
    research_report_openrouter_api_key: str | None = Field(default=None, validation_alias="RESEARCH_REPORT_OPENROUTER_API_KEY")
    research_report_openrouter_base_url: str | None = Field(default=None, validation_alias="RESEARCH_REPORT_OPENROUTER_BASE_URL")
    use_live_crews: bool = Field(default=False, validation_alias="USE_LIVE_CREWS")
    verbose_crews: bool = Field(default=False, validation_alias="VERBOSE_CREWS")
    default_sections: list[str] = Field(
        default_factory=lambda: [
            "Executive Summary",
            "Research Objective",
            "Key Questions",
            "Findings by Theme",
            "Evidence Table",
            "Risks / Conflicts / Unknowns",
            "Recommendations",
            "Source Appendix",
        ]
    )
    default_depth: str = "standard"
    web_search_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    tavily_search_depth: str = Field(default="basic", validation_alias="TAVILY_SEARCH_DEPTH")
    scholar_api_key: str | None = Field(default=None, validation_alias="SERPAPI_API_KEY")
    pdf_template_model_id: str = "opendatalab/MinerU2.5-Pro-2604-1.2B"
    pdf_template_max_pages: int = 6
    search_result_limit: int = Field(default=5, validation_alias="SEARCH_RESULT_LIMIT")
    scholar_result_limit: int = Field(default=5, validation_alias="SCHOLAR_RESULT_LIMIT")
    max_agenda_items: int = Field(default=0, validation_alias="MAX_AGENDA_ITEMS")
    dual_research_parallel: bool = Field(default=True, validation_alias="DUAL_RESEARCH_PARALLEL")
    request_timeout_seconds: float = 20.0
    api_retry_attempts: int = Field(default=3, validation_alias="API_RETRY_ATTEMPTS")
    api_retry_backoff_seconds: float = Field(default=1.0, validation_alias="API_RETRY_BACKOFF_SECONDS")
    llm_min_interval_seconds: float = Field(default=0.0, validation_alias="LLM_MIN_INTERVAL_SECONDS")
    fast_degrade_on_live_failure: bool = Field(default=True, validation_alias="FAST_DEGRADE_ON_LIVE_FAILURE")
    allow_live_fallback: bool = Field(default=False, validation_alias="ALLOW_LIVE_FALLBACK")
    free_api_mode: bool = Field(default=False, validation_alias="FREE_API_MODE")
    user_agent: str = "research-report-flow/0.1"

    def __init__(self, **values: object) -> None:
        field_aliases = {
            "llm_model": "RESEARCH_REPORT_MODEL",
            "research_report_gemini_api_key": "RESEARCH_REPORT_GEMINI_API_KEY",
            "research_report_openrouter_api_key": "RESEARCH_REPORT_OPENROUTER_API_KEY",
            "research_report_openrouter_base_url": "RESEARCH_REPORT_OPENROUTER_BASE_URL",
            "use_live_crews": "USE_LIVE_CREWS",
            "verbose_crews": "VERBOSE_CREWS",
            "web_search_api_key": "TAVILY_API_KEY",
            "tavily_search_depth": "TAVILY_SEARCH_DEPTH",
            "scholar_api_key": "SERPAPI_API_KEY",
            "search_result_limit": "SEARCH_RESULT_LIMIT",
            "scholar_result_limit": "SCHOLAR_RESULT_LIMIT",
            "max_agenda_items": "MAX_AGENDA_ITEMS",
            "dual_research_parallel": "DUAL_RESEARCH_PARALLEL",
            "api_retry_attempts": "API_RETRY_ATTEMPTS",
            "api_retry_backoff_seconds": "API_RETRY_BACKOFF_SECONDS",
            "llm_min_interval_seconds": "LLM_MIN_INTERVAL_SECONDS",
            "fast_degrade_on_live_failure": "FAST_DEGRADE_ON_LIVE_FAILURE",
            "allow_live_fallback": "ALLOW_LIVE_FALLBACK",
            "free_api_mode": "FREE_API_MODE",
        }
        for field_name, alias in field_aliases.items():
            if field_name in values and alias not in values:
                values[alias] = values.pop(field_name)
        super().__init__(**values)

    def should_use_live_crews(self) -> bool:
        if not self.use_live_crews:
            return False
        if self.llm_model is None:
            return False
        return self._has_live_llm_provider()

    def _has_live_llm_provider(self) -> bool:
        llm_signals = (
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("AZURE_OPENAI_API_KEY"),
            self.research_report_gemini_api_key,
            self.research_report_openrouter_api_key,
        )
        return any(llm_signals) or self._is_local_ollama_configuration()

    def _is_local_ollama_configuration(self) -> bool:
        if not self.llm_model or not self.llm_model.startswith("ollama/"):
            return False
        if not self.research_report_openrouter_base_url:
            return False
        parsed = urlparse(self.research_report_openrouter_base_url)
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    configure_runtime_environment()
    settings = Settings()
    if settings.research_report_gemini_api_key and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.research_report_gemini_api_key
    if settings.research_report_openrouter_api_key and not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = settings.research_report_openrouter_api_key
    if settings.research_report_openrouter_base_url:
        for env_name in ("OPENROUTER_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
            if not os.getenv(env_name):
                os.environ[env_name] = settings.research_report_openrouter_base_url
    if not settings.web_search_api_key and os.getenv("SERPER_API_KEY"):
        settings.web_search_api_key = os.getenv("SERPER_API_KEY")
    if settings.free_api_mode:
        settings.api_retry_attempts = max(settings.api_retry_attempts, 8)
        settings.api_retry_backoff_seconds = max(settings.api_retry_backoff_seconds, 2.0)
        settings.llm_min_interval_seconds = max(settings.llm_min_interval_seconds, 13.0)
        settings.search_result_limit = min(settings.search_result_limit, 3)
        settings.scholar_result_limit = min(settings.scholar_result_limit, 3)
        settings.max_agenda_items = 2 if settings.max_agenda_items <= 0 else min(settings.max_agenda_items, 2)
        settings.dual_research_parallel = False
    settings.output_root.mkdir(parents=True, exist_ok=True)
    return settings
