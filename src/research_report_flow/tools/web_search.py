from __future__ import annotations

import json
from typing import Callable

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from ..config import Settings, get_settings
from ..models import CandidateSource
from ..utils import retry_call


class WebSearchArgs(BaseModel):
    query: str
    max_results: int = 5
    recency_days: int | None = None


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the open web for sources relevant to the research question."
    args_schema: type[BaseModel] = WebSearchArgs
    _settings: Settings = PrivateAttr()
    _search_impl: Callable[[str, int, int | None], list[CandidateSource]] | None = PrivateAttr(default=None)

    def __init__(
        self,
        settings: Settings | None = None,
        search_impl: Callable[[str, int, int | None], list[CandidateSource]] | None = None,
        **data: object,
    ) -> None:
        super().__init__(**data)
        self._settings = settings or get_settings()
        self._search_impl = search_impl

    def search(self, query: str, max_results: int | None = None, recency_days: int | None = None) -> list[CandidateSource]:
        if self._search_impl is not None:
            return self._search_impl(query, max_results or self._settings.search_result_limit, recency_days)

        if not self._settings.should_use_live_crews() or not self._settings.web_search_api_key:
            slug = "-".join(query.lower().split())[:40] or "query"
            return [
                CandidateSource(
                    source_id="web-fallback-1",
                    title=f"Fallback web source for {query}",
                    url=f"https://example.com/web/{slug}",
                    source_type="web",
                    query=query,
                    snippet="Deterministic fallback result used because no web search API key is configured.",
                )
            ]

        payload: dict[str, object] = {
            "query": query,
            "max_results": max_results or self._settings.search_result_limit,
            "search_depth": self._settings.tavily_search_depth,
        }
        if recency_days is not None:
            payload["time_range"] = _map_time_range(recency_days)

        headers = {
            "Authorization": f"Bearer {self._settings.web_search_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self._settings.request_timeout_seconds, headers={"User-Agent": self._settings.user_agent}) as client:
            def _request() -> dict[str, object]:
                response = client.post("https://api.tavily.com/search", headers=headers, json=payload)
                response.raise_for_status()
                return response.json()

            data = retry_call(
                _request,
                attempts=self._settings.api_retry_attempts,
                base_delay_seconds=self._settings.api_retry_backoff_seconds,
            )

        results: list[CandidateSource] = []
        result_items = data.get("results", [])
        if not isinstance(result_items, list):
            result_items = []
        for index, item in enumerate(result_items[: int(payload["max_results"])], start=1):
            if not isinstance(item, dict):
                continue
            results.append(
                CandidateSource(
                    source_id=f"web-{index}",
                    title=item.get("title") or f"Web Result {index}",
                    url=item.get("url", ""),
                    source_type="web",
                    query=query,
                    snippet=item.get("content", ""),
                )
            )
        return results

    def _run(self, query: str, max_results: int = 5, recency_days: int | None = None) -> str:
        return json.dumps(
            [source.model_dump(mode="json") for source in self.search(query, max_results, recency_days)],
            ensure_ascii=False,
        )


def _map_time_range(recency_days: int) -> str:
    if recency_days <= 1:
        return "day"
    if recency_days <= 7:
        return "week"
    if recency_days <= 31:
        return "month"
    return "year"
