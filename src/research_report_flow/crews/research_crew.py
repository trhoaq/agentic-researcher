from __future__ import annotations

from ..config import Settings, get_settings
from ..models import CandidateSource, SearchAgendaItem
from ..tools import WebSearchTool
from .base import build_agenda_query, log_degrade, normalize_candidate_batch, should_degrade_to_fallback


class ResearchCrew:
    def __init__(self, settings: Settings | None = None, tool: WebSearchTool | None = None) -> None:
        self.settings = settings or get_settings()
        self.tool = tool or WebSearchTool(settings=self.settings)

    def run(self, agenda: list[SearchAgendaItem]) -> list[CandidateSource]:
        if not self.settings.should_use_live_crews():
            return self._fallback_search(agenda)

        try:
            sources: list[CandidateSource] = []
            for agenda_index, item in enumerate(agenda, start=1):
                query = build_agenda_query(item)
                raw_sources = self.tool.search(
                    query=query,
                    max_results=self.settings.search_result_limit,
                    recency_days=item.recency_days,
                )
                sources.extend(
                    normalize_candidate_batch(
                        raw_sources,
                        source_type="web",
                        query=query,
                        source_id_prefix=f"web-{agenda_index}",
                    )
                )
            return sources
        except (TypeError, ValueError) as exc:
            if self.settings.allow_live_fallback:
                log_degrade(exc, stage="web_research_normalize")
                return self._fallback_search(agenda)
            raise
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="web_research")
                return self._fallback_search(agenda)
            raise

    def _fallback_search(self, agenda: list[SearchAgendaItem]) -> list[CandidateSource]:
        sources: list[CandidateSource] = []
        for index, item in enumerate(agenda, start=1):
            query = build_agenda_query(item)
            slug = "-".join(query.lower().split())[:40] or f"query-{index}"
            sources.append(
                CandidateSource(
                    source_id=f"web-fallback-{index}",
                    title=f"Fallback web source for {query}",
                    url=f"https://example.com/web/{slug}",
                    source_type="web",
                    query=query,
                    snippet="Deterministic fallback result used after live web-research failure.",
                )
            )
        return sources
