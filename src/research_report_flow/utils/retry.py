from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

import httpx


T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_MARKERS = (
    "429",
    "503",
    "502",
    "504",
    "provider returned error",
    "rate-limited upstream",
    "rate-limited",
    "resource_exhausted",
    "quota exceeded",
    "exceeded your current quota",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "connection error",
    "failed to connect",
    "'nonetype' object is not subscriptable",
    "nonetype object is not subscriptable",
    "object is not subscriptable",
)


def should_retry_exception(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        return response is not None and response.status_code in RETRYABLE_STATUS_CODES

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True

    lowered = str(exc).lower()
    return any(marker in lowered for marker in RETRYABLE_ERROR_MARKERS)


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    message = str(exc)
    patterns = (
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retrydelay['\"]?\s*[:=]\s*['\"]([0-9]+(?:\.[0-9]+)?)s",
        r"retry delay['\"]?\s*[:=]\s*['\"]([0-9]+(?:\.[0-9]+)?)s",
    )
    lowered = message.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float = 8.0,
    should_retry: Callable[[Exception], bool] = should_retry_exception,
) -> T:
    if attempts <= 1:
        return operation()

    delay = max(base_delay_seconds, 0.0)
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            dynamic_delay = _extract_retry_delay_seconds(exc)
            effective_delay = delay
            if dynamic_delay is not None:
                effective_delay = max(effective_delay, dynamic_delay)
            if effective_delay > 0:
                time.sleep(effective_delay)
            delay = min(max(effective_delay, max(base_delay_seconds, 0.5)) * 2, max_delay_seconds)

    return operation()
