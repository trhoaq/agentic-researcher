from __future__ import annotations

import json
from typing import Callable

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from ..config import Settings, get_settings
from ..models import CandidateSource
from ..utils import retry_call


class GoogleScholarArgs(BaseModel):
    query: str
    max_results: int = 5
    year_from: int | None = None


class GoogleScholarSearchTool(BaseTool):
    name: str = "google_scholar_search"
    description: str = "Search scholarly sources via a Google Scholar-compatible provider."
    args_schema: type[BaseModel] = GoogleScholarArgs
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

    def search(self, query: str, max_results: int | None = None, year_from: int | None = None) -> list[CandidateSource]:
        if self._search_impl is not None:
            return self._search_impl(query, max_results or self._settings.scholar_result_limit, year_from)

        if not self._settings.should_use_live_crews() or not self._settings.scholar_api_key:
            slug = "-".join(query.lower().split())[:40] or "query"
            return [
                CandidateSource(
                    source_id="scholar-fallback-1",
                    title=f"Fallback scholar source for {query}",
                    url=f"https://example.com/scholar/{slug}",
                    source_type="scholar",
                    query=query,
                    snippet="Deterministic fallback result used because no scholar API key is configured.",
                    authors=["Fallback Researcher"],
                    year=2024,
                )
            ]

        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self._settings.scholar_api_key,
            "num": max_results or self._settings.scholar_result_limit,
        }
        if year_from is not None:
            params["as_ylo"] = year_from

        with httpx.Client(timeout=self._settings.request_timeout_seconds, headers={"User-Agent": self._settings.user_agent}) as client:
            def _request() -> dict[str, object]:
                response = client.get("https://serpapi.com/search.json", params=params)
                response.raise_for_status()
                return response.json()

            data = retry_call(
                _request,
                attempts=self._settings.api_retry_attempts,
                base_delay_seconds=self._settings.api_retry_backoff_seconds,
            )

        results: list[CandidateSource] = []
        for index, item in enumerate(data.get("organic_results", [])[: params["num"]], start=1):
            publication = item.get("publication_info", {}) or {}
            authors = [author.get("name", "") for author in publication.get("authors", []) if author.get("name")]
            results.append(
                CandidateSource(
                    source_id=f"scholar-{index}",
                    title=item.get("title") or f"Scholar Result {index}",
                    url=item.get("link") or item.get("resources", [{}])[0].get("link", ""),
                    source_type="scholar",
                    query=query,
                    snippet=item.get("snippet", ""),
                    authors=authors,
                    venue=publication.get("summary"),
                    year=_extract_year(publication.get("summary")),
                )
            )
        return results

    def _run(self, query: str, max_results: int = 5, year_from: int | None = None) -> str:
        return json.dumps(
            [source.model_dump(mode="json") for source in self.search(query, max_results, year_from)],
            ensure_ascii=False,
        )


def _extract_year(summary: str | None) -> int | None:
    if not summary:
        return None
    for token in summary.replace(",", " ").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None
