from __future__ import annotations

import json
import re
from typing import Callable
from urllib.parse import urlparse

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from ..config import Settings, get_settings
from ..models import FetchedSource
from ..utils import retry_call


class SourceFetchArgs(BaseModel):
    url: str


class SourceFetcher:
    def __init__(
        self,
        settings: Settings | None = None,
        fetch_impl: Callable[[str], FetchedSource] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._fetch_impl = fetch_impl

    def fetch(self, url: str) -> FetchedSource:
        if self._fetch_impl is not None:
            return self._fetch_impl(url)

        parsed = urlparse(url)
        if parsed.netloc.lower() == "example.com":
            return FetchedSource(
                url=url,
                final_url=url,
                status_code=200,
                title=parsed.path.strip("/").replace("-", " ").title() or "Example Source",
                excerpt="Deterministic fallback source content.",
            )

        if not self.settings.should_use_live_crews():
            return FetchedSource(
                url=url,
                final_url=url,
                status_code=200,
                title=parsed.path.strip("/").replace("-", " ").title() or "Fallback Source",
                excerpt="Deterministic fallback source content.",
            )

        with httpx.Client(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.settings.user_agent},
        ) as client:
            try:
                def _request() -> httpx.Response:
                    response = client.get(url)
                    response.raise_for_status()
                    return response

                response = retry_call(
                    _request,
                    attempts=self.settings.api_retry_attempts,
                    base_delay_seconds=self.settings.api_retry_backoff_seconds,
                )
            except httpx.HTTPError:
                if not self.settings.should_use_live_crews():
                    return FetchedSource(
                        url=url,
                        final_url=url,
                        status_code=200,
                        title=parsed.path.strip("/").replace("-", " ").title() or "Fallback Source",
                        excerpt="Deterministic fallback source content.",
                    )
                raise

        excerpt = re.sub(r"\s+", " ", response.text)[:500]
        title_match = re.search(r"<title>(.*?)</title>", response.text, flags=re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else None
        return FetchedSource(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            title=title,
            excerpt=excerpt,
        )


class SourceFetchTool(BaseTool):
    name: str = "fetch_source"
    description: str = "Fetch source metadata and a short excerpt from a URL."
    args_schema: type[BaseModel] = SourceFetchArgs
    _fetcher: SourceFetcher = PrivateAttr()

    def __init__(self, fetcher: SourceFetcher | None = None, **data: object) -> None:
        super().__init__(**data)
        self._fetcher = fetcher or SourceFetcher()

    def _run(self, url: str) -> str:
        return json.dumps(self._fetcher.fetch(url).model_dump(mode="json"), ensure_ascii=False)
