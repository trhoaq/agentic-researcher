from __future__ import annotations

import pytest

from research_report_flow.config import Settings
from research_report_flow.crews.research_crew import ResearchCrew
from research_report_flow.crews.scholar_research_crew import ScholarResearchCrew
from research_report_flow.models import SearchAgendaItem
from research_report_flow.tools.google_scholar_search import GoogleScholarSearchTool
from research_report_flow.tools.web_search import WebSearchTool


def test_research_fallback_search_is_deterministic_without_network_tool_calls() -> None:
    settings = Settings(USE_LIVE_CREWS=True, RESEARCH_REPORT_MODEL="openrouter/openai/gpt-oss-20b:free")
    web_tool = WebSearchTool(settings=settings, search_impl=lambda *_: (_ for _ in ()).throw(RuntimeError("should not call tool")))
    crew = ResearchCrew(settings=settings, tool=web_tool)
    sources = crew._fallback_search([SearchAgendaItem(question="What changed?", keywords=["ai"])])
    assert len(sources) == 1
    assert sources[0].source_id.startswith("web-fallback-")


def test_scholar_fallback_search_is_deterministic_without_network_tool_calls() -> None:
    settings = Settings(USE_LIVE_CREWS=True, RESEARCH_REPORT_MODEL="openrouter/openai/gpt-oss-20b:free")
    scholar_tool = GoogleScholarSearchTool(
        settings=settings, search_impl=lambda *_: (_ for _ in ()).throw(RuntimeError("should not call tool"))
    )
    crew = ScholarResearchCrew(settings=settings, tool=scholar_tool)
    sources = crew._fallback_search([SearchAgendaItem(question="What changed?", keywords=["ai"])])
    assert len(sources) == 1
    assert sources[0].source_id.startswith("scholar-fallback-")


@pytest.mark.parametrize("raw_query", [None, "", "provided query"])
def test_research_live_search_normalizes_query_and_unique_source_ids(raw_query: str | None) -> None:
    settings = Settings(
        USE_LIVE_CREWS=True,
        RESEARCH_REPORT_MODEL="openrouter/openai/gpt-oss-20b:free",
        RESEARCH_REPORT_OPENROUTER_API_KEY="test-key",
    )
    crew = ResearchCrew(
        settings=settings,
        tool=WebSearchTool(
            settings=settings,
            search_impl=lambda query, max_results, recency_days: [
                {
                    "source_id": "raw-web",
                    "title": "Open Web Source",
                    "url": "https://example.com/web",
                    "source_type": "scholar",
                    "query": raw_query,
                    "snippet": "web snippet",
                }
            ],
        ),
    )

    sources = crew.run([SearchAgendaItem(question="What changed?", keywords=["ai"])])

    assert len(sources) == 1
    expected_query = raw_query or "What changed? ai"
    assert sources[0].query == expected_query
    assert sources[0].source_id == "web-1-1"
    assert sources[0].source_type == "web"


@pytest.mark.parametrize("raw_query", [None, "", "provided query"])
def test_scholar_live_search_normalizes_query_and_unique_source_ids(raw_query: str | None) -> None:
    settings = Settings(
        USE_LIVE_CREWS=True,
        RESEARCH_REPORT_MODEL="openrouter/openai/gpt-oss-20b:free",
        RESEARCH_REPORT_OPENROUTER_API_KEY="test-key",
    )
    crew = ScholarResearchCrew(
        settings=settings,
        tool=GoogleScholarSearchTool(
            settings=settings,
            search_impl=lambda query, max_results, year_from: [
                {
                    "source_id": "raw-scholar",
                    "title": "Scholar Source",
                    "url": "https://example.com/paper",
                    "source_type": "web",
                    "query": raw_query,
                    "snippet": "paper snippet",
                    "authors": ["A. Researcher"],
                    "year": 2024,
                }
            ],
        ),
    )

    sources = crew.run([SearchAgendaItem(question="What changed?", keywords=["ai"], scholar_year_from=2020)])

    assert len(sources) == 1
    expected_query = raw_query or "What changed? ai"
    assert sources[0].query == expected_query
    assert sources[0].source_id == "scholar-1-1"
    assert sources[0].source_type == "scholar"
