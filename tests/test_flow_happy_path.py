from __future__ import annotations

import shutil
from pathlib import Path

from research_report_flow.config import Settings
from research_report_flow.flow import ResearchReportFlow
from research_report_flow.models import CandidateSource, FetchedSource
from research_report_flow.tools.fetch_source import SourceFetcher
from research_report_flow.tools.google_scholar_search import GoogleScholarSearchTool
from research_report_flow.tools.web_search import WebSearchTool


def test_flow_happy_path() -> None:
    tmp_path = Path("F:/Code/agent/.tmp/test-flow-happy")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(project_root=tmp_path, output_root=tmp_path / "output", use_live_crews=False)

    web_tool = WebSearchTool(
        settings=settings,
        search_impl=lambda query, max_results, recency_days: [
            CandidateSource(
                source_id="web-1",
                title="Open Web Source",
                url="https://example.com/web",
                source_type="web",
                query=query,
                snippet="web snippet",
            )
        ],
    )
    scholar_tool = GoogleScholarSearchTool(
        settings=settings,
        search_impl=lambda query, max_results, year_from: [
            CandidateSource(
                source_id="scholar-1",
                title="Scholar Source",
                url="https://example.com/paper",
                source_type="scholar",
                query=query,
                snippet="paper snippet",
                authors=["A. Researcher"],
                year=2024,
            )
        ],
    )
    fetcher = SourceFetcher(
        settings=settings,
        fetch_impl=lambda url: FetchedSource(url=url, final_url=url, status_code=200, title="Fetched", excerpt="body"),
    )

    from research_report_flow.crews import ResearchCrew, ScholarResearchCrew, VerificationCrew

    flow = ResearchReportFlow(
        settings=settings,
        research_crew=ResearchCrew(settings=settings, tool=web_tool),
        scholar_research_crew=ScholarResearchCrew(settings=settings, tool=scholar_tool),
        verification_crew=VerificationCrew(settings=settings, fetcher=fetcher),
    )

    flow.kickoff(
        inputs={
            "brief": "# AI Agents\nObjective: summarize current patterns\nWhat are the key risks?",
            "format_instructions": "- Executive Summary\n- Findings by Theme\n- Source Appendix",
            "output_dir": str(tmp_path / "run"),
        }
    )

    final_path = Path(flow.state.final_report_path or "")
    assert final_path.exists()
    markdown = final_path.read_text(encoding="utf-8")
    assert "## Executive Summary" in markdown
    assert "## Findings by Theme" in markdown
    assert "## Source Appendix" in markdown
    assert len(flow.state.web_candidates) == 1
    assert len(flow.state.scholar_candidates) == 1
    assert len(flow.state.verified_sources) == 2
