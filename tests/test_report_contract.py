from __future__ import annotations

from research_report_flow.config import Settings
from research_report_flow.crews.refinement_crew import RefinementCrew
from research_report_flow.crews.writing_crew import WritingCrew
from research_report_flow.models import EvidenceSynthesis, ReportFormatSpec, ReportRequirements, ThematicFinding, VerifiedSource


def test_writer_respects_format_sections() -> None:
    settings = Settings(project_root="F:/Code/agent", output_root="F:/Code/agent/output", use_live_crews=False)
    writer = WritingCrew(settings=settings)
    report = writer.run(
        requirements=ReportRequirements(topic="AI agents"),
        format_spec=ReportFormatSpec(sections=["Summary", "Source Appendix"]),
        outline=None,
        synthesis=EvidenceSynthesis(findings=[ThematicFinding(theme="Summary", summary="Done")]),
        sources=[
            VerifiedSource(
                source_id="s1",
                title="Example",
                url="https://example.com",
                normalized_url="https://example.com/",
                source_type="web",
                query="ai agents",
                citation_label="[1] Example",
            )
        ],
    )
    assert "## Summary" in report.markdown
    assert "## Source Appendix" in report.markdown


def test_refinement_inserts_missing_section() -> None:
    settings = Settings(project_root="F:/Code/agent", output_root="F:/Code/agent/output", use_live_crews=False)
    refiner = RefinementCrew(settings=settings)
    result = refiner.run(
        requirements=ReportRequirements(topic="AI agents"),
        format_spec=ReportFormatSpec(sections=["Executive Summary", "Recommendations"]),
        synthesis=EvidenceSynthesis(),
        draft=type("Draft", (), {"markdown": "## Executive Summary\nReady\n"})(),
    )
    assert "## Recommendations" in result.markdown
