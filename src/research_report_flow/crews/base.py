from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Literal, TypeVar

from crewai import Crew
from pydantic import BaseModel

from ..config import Settings
from ..models import CandidateSource, EvidenceSynthesis, SearchAgendaItem
from ..utils import retry_call, should_retry_exception

try:
    import yaml
except Exception:  # pragma: no cover - fallback for minimal environments
    yaml = None


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
_JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.IGNORECASE | re.DOTALL)


def parse_structured_output(result: object, model: type[T]) -> T:
    if not any(hasattr(result, attr) for attr in ("pydantic", "json_dict", "raw")):
        return model.model_validate(result)

    pydantic_value = getattr(result, "pydantic", None)
    if pydantic_value is not None:
        return model.model_validate(pydantic_value)

    payload = extract_json_result_payload(result, context=model.__name__)
    return model.model_validate(coerce_structured_payload(payload, model))


def extract_json_result_payload(result: object, *, context: str) -> object:
    json_value = getattr(result, "json_dict", None)
    if json_value is not None:
        return json_value

    raw_value = getattr(result, "raw", None)
    if raw_value is not None:
        return parse_json_maybe_embedded(raw_value, context=context)

    raise ValueError(f"{context} result has no raw/json payload")


def parse_json_maybe_embedded(raw_value: object, *, context: str) -> object:
    if not isinstance(raw_value, str):
        return raw_value

    text = raw_value.strip()
    if not text:
        raise ValueError(f"{context} result returned empty text instead of JSON")

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_parse_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        if yaml is not None:
            try:
                loaded = yaml.safe_load(candidate)
            except Exception:
                loaded = None
            if loaded is not None and not isinstance(loaded, str):
                return loaded

    raise ValueError(f"{context} result did not contain valid JSON") from last_error


def _json_parse_candidates(text: str) -> list[str]:
    candidates: list[str] = [text]

    fenced_match = _JSON_CODE_FENCE_RE.match(text)
    if fenced_match is not None:
        candidates.append(fenced_match.group("body").strip())

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1].strip())

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def build_agenda_query(item: SearchAgendaItem) -> str:
    return " ".join([item.question, *item.keywords]).strip()


def normalize_candidate_batch(
    raw_sources: list[object],
    *,
    source_type: Literal["web", "scholar"],
    query: str,
    source_id_prefix: str,
) -> list[CandidateSource]:
    normalized: list[CandidateSource] = []
    for index, raw_source in enumerate(raw_sources, start=1):
        payload = _candidate_payload(raw_source)
        payload["source_id"] = f"{source_id_prefix}-{index}"
        payload["source_type"] = source_type
        if not payload.get("query"):
            payload["query"] = query
        normalized.append(CandidateSource.model_validate(payload))
    return normalized


def _candidate_payload(raw_source: object) -> dict[str, object]:
    if isinstance(raw_source, CandidateSource):
        return raw_source.model_dump(mode="json")
    if isinstance(raw_source, BaseModel):
        return raw_source.model_dump(mode="json")
    if isinstance(raw_source, dict):
        return dict(raw_source)
    raise TypeError(f"unsupported candidate source payload type: {type(raw_source).__name__}")


def coerce_structured_payload(payload: object, model: type[T]) -> object:
    if model is EvidenceSynthesis and isinstance(payload, dict):
        return _coerce_evidence_synthesis_payload(payload)
    return payload


def _coerce_evidence_synthesis_payload(payload: dict[str, object]) -> dict[str, object]:
    evidence_rows = payload.get("evidence_rows", payload.get("evidence", []))
    findings = payload.get("findings", payload.get("themes", payload.get("thematic_findings", [])))
    unknowns = payload.get("unknowns", payload.get("open_questions", []))
    return {
        "evidence_rows": [_coerce_evidence_row(item) for item in _ensure_list(evidence_rows) if isinstance(item, dict)],
        "findings": [_coerce_thematic_finding(item) for item in _ensure_list(findings) if isinstance(item, dict)],
        "unknowns": [str(item) for item in _ensure_list(unknowns) if item not in (None, "")],
    }


def _coerce_evidence_row(item: dict[str, object]) -> dict[str, object]:
    source_ids = _coerce_source_ids(
        item.get("source_ids", item.get("sources", item.get("source", item.get("source_id"))))
    )
    claim = (
        item.get("claim")
        or item.get("title")
        or item.get("statement")
        or item.get("summary")
        or item.get("source")
        or "Untitled evidence"
    )
    return {
        "claim": str(claim),
        "source_ids": source_ids,
        "confidence": str(item.get("confidence", "medium")),
        "notes": str(item.get("notes", item.get("summary", item.get("detail", "")))),
    }


def _coerce_thematic_finding(item: dict[str, object]) -> dict[str, object]:
    evidence_source_ids = _coerce_source_ids(
        item.get("evidence_source_ids", item.get("source_ids", item.get("sources", item.get("source"))))
    )
    theme = item.get("theme") or item.get("title") or item.get("name") or "Untitled theme"
    summary = item.get("summary") or item.get("finding") or item.get("description") or ""
    open_questions = item.get("open_questions", item.get("questions", item.get("unknowns", [])))
    return {
        "theme": str(theme),
        "summary": str(summary),
        "evidence_source_ids": evidence_source_ids,
        "open_questions": [str(question) for question in _ensure_list(open_questions) if question not in (None, "")],
    }


def _coerce_source_ids(value: object) -> list[str]:
    return [str(item) for item in _ensure_list(value) if item not in (None, "")]


def _ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
