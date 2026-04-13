from __future__ import annotations

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
