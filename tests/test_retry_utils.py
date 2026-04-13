from __future__ import annotations

import httpx

from research_report_flow.utils.retry import retry_call, should_retry_exception


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(f"status {status_code}", request=request, response=response)


def test_retry_call_retries_on_503_then_succeeds() -> None:
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _http_status_error(503)
        return "ok"

    assert retry_call(operation, attempts=3, base_delay_seconds=0.0) == "ok"
    assert attempts["count"] == 3


def test_retry_call_does_not_retry_on_400() -> None:
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        raise _http_status_error(400)

    try:
        retry_call(operation, attempts=3, base_delay_seconds=0.0)
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 400
    else:
        raise AssertionError("Expected HTTPStatusError for 400 response")

    assert attempts["count"] == 1


def test_should_retry_exception_recognizes_service_unavailable_message() -> None:
    assert should_retry_exception(RuntimeError("Service Unavailable: 503")) is True


def test_should_retry_exception_recognizes_quota_exhausted_message() -> None:
    message = "429 RESOURCE_EXHAUSTED. You exceeded your current quota. Please retry in 0s."
    assert should_retry_exception(RuntimeError(message)) is True
