from __future__ import annotations

from research_report_flow.crews.base import should_degrade_to_fallback


def test_should_degrade_to_fallback_on_openrouter_rate_limit_message() -> None:
    message = "OpenAI API call failed: Error code: 429 - {'error': {'message': 'Provider returned error', 'metadata': {'raw': 'openai/gpt-oss-20b:free is temporarily rate-limited upstream. Please retry shortly'}}}"
    assert should_degrade_to_fallback(RuntimeError(message)) is True


def test_should_degrade_to_fallback_on_openai_connection_error_message() -> None:
    assert should_degrade_to_fallback(ConnectionError("Failed to connect to OpenAI API: Connection error.")) is True


def test_should_degrade_to_fallback_on_nonetype_subscriptable_message() -> None:
    assert should_degrade_to_fallback(RuntimeError("'NoneType' object is not subscriptable")) is True
