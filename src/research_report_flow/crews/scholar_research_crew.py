from __future__ import annotations

import json
from typing import Any

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import CandidateSource, SearchAgendaItem
from ..tools import GoogleScholarSearchTool
from .base import kickoff_with_retry, log_degrade, should_degrade_to_fallback


class ScholarResearchCrew:
    def __init__(self, settings: Settings | None = None, tool: GoogleScholarSearchTool | None = None) -> None:
        self.settings = settings or get_settings()
        self.tool = tool or GoogleScholarSearchTool(settings=self.settings)

    def run(self, agenda: list[SearchAgendaItem]) -> list[CandidateSource]:
        if not self.settings.should_use_live_crews():
            return self._fallback_search(agenda)

        try:
            agent = Agent(
                role="Scholar Researcher",
                goal="Collect scholarly sources relevant to the research agenda.",
                backstory="You prioritize papers, venues, abstracts, and citation-ready metadata.",
                llm=self.settings.llm_model,
                tools=[self.tool],
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Use the google_scholar_search tool to gather scholarly sources for each agenda item. Return a JSON list "
                    "of CandidateSource objects.\n\nAgenda:\n{agenda_json}"
                ),
                expected_output="A JSON array of CandidateSource objects.",
                agent=agent,
                markdown=False,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={"agenda_json": json.dumps([item.model_dump(mode="json") for item in agenda], ensure_ascii=False, indent=2)},
                settings=self.settings,
            )
            parsed_items = _extract_candidate_source_items(result)
            return [CandidateSource.model_validate(item) for item in parsed_items]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if self.settings.allow_live_fallback:
                log_degrade(exc, stage="scholar_research_parse")
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
            query = " ".join([item.question, *item.keywords]).strip()
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


def _extract_candidate_source_items(result: object) -> list[dict[str, Any]]:
    json_value = getattr(result, "json_dict", None)
    if isinstance(json_value, list):
        return [item for item in json_value if isinstance(item, dict)]

    raw_value = getattr(result, "raw", None)
    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]

    if isinstance(raw_value, str):
        loaded = json.loads(raw_value)
    elif raw_value is not None:
        loaded = raw_value
    else:
        raise ValueError("scholar crew result has no raw/json payload")

    if not isinstance(loaded, list):
        raise ValueError("scholar crew output is not a JSON list")
    return [item for item in loaded if isinstance(item, dict)]
