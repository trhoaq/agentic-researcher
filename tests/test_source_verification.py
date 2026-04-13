from __future__ import annotations

from research_report_flow.config import Settings
from research_report_flow.crews.verification_crew import VerificationCrew
from research_report_flow.models import CandidateSource, FetchedSource
from research_report_flow.tools.fetch_source import SourceFetcher


def test_verification_deduplicates_normalized_urls() -> None:
    settings = Settings(project_root="F:/Code/agent", output_root="F:/Code/agent/output", use_live_crews=False)
    fetcher = SourceFetcher(
        settings=settings,
        fetch_impl=lambda url: FetchedSource(url=url, final_url=url, status_code=200, title="Fetched", excerpt="ok"),
    )
    crew = VerificationCrew(settings=settings, fetcher=fetcher)
    verified = crew.run(
        [
            CandidateSource(source_id="a", title="One", url="https://example.com/path/?b=2&a=1", source_type="web", query="q"),
            CandidateSource(source_id="b", title="Two", url="https://example.com/path/?a=1&b=2", source_type="web", query="q"),
        ]
    )
    assert len(verified) == 1
    assert verified[0].citation_label.startswith("[1]")
