from __future__ import annotations

from ..config import Settings, get_settings
from ..models import CandidateSource, SearchAgendaItem
from ..tools import GoogleScholarSearchTool
from .base import build_agenda_query, log_degrade, normalize_candidate_batch, should_degrade_to_fallback


class ScholarResearchCrew:
    def __init__(self, settings: Settings | None = None, tool: GoogleScholarSearchTool | None = None) -> None:
        self.settings = settings or get_settings()
        self.tool = tool or GoogleScholarSearchTool(settings=self.settings)

    def run(self, agenda: list[SearchAgendaItem]) -> list[CandidateSource]:
        if not self.settings.should_use_live_crews():
            return self._fallback_search(agenda)

        try:
            sources: list[CandidateSource] = []
            for agenda_index, item in enumerate(agenda, start=1):
                query = build_agenda_query(item)
                raw_sources = self.tool.search(
                    query=query,
                    max_results=self.settings.scholar_result_limit,
                    year_from=item.scholar_year_from,
                )
                sources.extend(
                    normalize_candidate_batch(
                        raw_sources,
                        source_type="scholar",
                        query=query,
                        source_id_prefix=f"scholar-{agenda_index}",
                    )
                )
            return sources
        except (TypeError, ValueError) as exc:
            if self.settings.allow_live_fallback:
                log_degrade(exc, stage="scholar_research_normalize")
                return self._fallback_search(agenda)
            raise
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="scholar_research")
                return self._fallback_search(agenda)
            raise

    def _fallback_search(self, agenda: list[SearchAgendaItem]) -> list[CandidateSource]:
        sources: list[CandidateSource] = []
        for index, item in enumerate(agenda, start=1):
            query = build_agenda_query(item)
            slug = "-".join(query.lower().split())[:40] or f"query-{index}"
            sources.append(
                CandidateSource(
                    source_id=f"scholar-fallback-{index}",
                    title=f"Fallback scholar source for {query}",
                    url=f"https://example.com/scholar/{slug}",
                    source_type="scholar",
                    query=query,
                    snippet="Deterministic fallback result used after live scholar-research failure.",
                    authors=["Fallback Researcher"],
                    year=2024,
                )
            )
        return sources
