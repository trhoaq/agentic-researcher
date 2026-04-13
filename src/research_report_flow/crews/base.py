from __future__ import annotations

import json
import logging
import threading
import time
from typing import TypeVar

from crewai import Crew
from pydantic import BaseModel

from ..config import Settings
from ..utils import retry_call, should_retry_exception


T = TypeVar("T", bound=BaseModel)
_LLM_CALL_LOCK = threading.Lock()
_NEXT_LLM_CALL_TS = 0.0
_LOGGER = logging.getLogger(__name__)
_FALLBACK_ERROR_MARKERS = (
    "openai api call failed",
    "google gemini api call failed",
    "failed to connect to openai api",
    "connection error",
    "'nonetype' object is not subscriptable",
    "nonetype object is not subscriptable",
    "object is not subscriptable",
    "authentication required",
    "invalid api key",
    "quota",
    "resource_exhausted",
    "rate-limit",
    "rate-limited",
    "provider returned error",
)


def parse_structured_output(result: object, model: type[T]) -> T:
    pydantic_value = getattr(result, "pydantic", None)
    if pydantic_value is not None:
        return model.model_validate(pydantic_value)

    json_value = getattr(result, "json_dict", None)
    if json_value is not None:
        return model.model_validate(json_value)

    raw_value = getattr(result, "raw", None)
    if raw_value is not None:
        if isinstance(raw_value, str):
            return model.model_validate(json.loads(raw_value))
        return model.model_validate(raw_value)

    return model.model_validate(result)


def kickoff_with_retry(crew: Crew, *, inputs: dict[str, object], settings: Settings) -> object:
    def _run_with_spacing() -> object:
        _wait_for_llm_slot(settings.llm_min_interval_seconds)
        return crew.kickoff(inputs=inputs)

    attempts = (
        1
        if settings.fast_degrade_on_live_failure and settings.allow_live_fallback
        else settings.api_retry_attempts
    )
    return retry_call(
        _run_with_spacing,
        attempts=attempts,
        base_delay_seconds=settings.api_retry_backoff_seconds,
    )


def should_degrade_to_fallback(exc: Exception) -> bool:
    if should_retry_exception(exc):
        return True
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _FALLBACK_ERROR_MARKERS)


def log_degrade(exc: Exception, stage: str) -> None:
    _LOGGER.warning("Degrading to deterministic fallback at stage '%s' due to live-provider failure: %s", stage, exc)


def _wait_for_llm_slot(min_interval_seconds: float) -> None:
    if min_interval_seconds <= 0:
        return
    global _NEXT_LLM_CALL_TS
    with _LLM_CALL_LOCK:
        now = time.monotonic()
        wait_seconds = _NEXT_LLM_CALL_TS - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        _NEXT_LLM_CALL_TS = now + min_interval_seconds
