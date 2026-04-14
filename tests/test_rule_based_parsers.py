from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

from research_report_flow.config import Settings
from research_report_flow.crews.format_crew import ReportFormatCrew
from research_report_flow.crews.pdf_template_crew import PdfTemplateCrew
import research_report_flow.crews.pdf_template_crew as pdf_template_crew_module
from research_report_flow.crews.requirements_crew import ReportRequirementsCrew
import research_report_flow.crews.requirements_crew as requirements_crew_module
from research_report_flow.models import PdfTemplateSample
from research_report_flow.tools import PdfTemplateParser


def test_requirements_parser_uses_llm_enrichment_in_live_mode(monkeypatch) -> None:
    settings = Settings(
        USE_LIVE_CREWS=True,
        RESEARCH_REPORT_MODEL="ollama/gemma4:31b-cloud",
        RESEARCH_REPORT_OPENROUTER_BASE_URL="http://localhost:11434/v1",
    )
    monkeypatch.setattr(
        requirements_crew_module,
        "kickoff_with_retry",
        lambda crew, inputs, settings: SimpleNamespace(
            json_dict={
                "topic": "AI Agents",
                "audience": "engineering leaders",
                "objectives": ["compare orchestration patterns"],
                "depth": "deep",
                "constraints": ["stay concise"],
                "must_answer_questions": ["What changed?"],
                "missing_information": ["Target deployment constraints"],
            }
        ),
    )

    parsed = ReportRequirementsCrew(settings=settings).run(
        "# AI Agents\nObjective: compare orchestration patterns\nConstraint: stay concise\nWhat changed?"
    )

    assert parsed.topic == "AI Agents"
    assert parsed.audience == "engineering leaders"
    assert parsed.depth == "deep"
    assert parsed.must_answer_questions == ["What changed?"]


def test_format_parser_stays_rule_based_in_live_mode() -> None:
    settings = Settings(
        USE_LIVE_CREWS=True,
        RESEARCH_REPORT_MODEL="ollama/gemma4:31b-cloud",
        RESEARCH_REPORT_OPENROUTER_BASE_URL="http://localhost:11434/v1",
    )
    pdf_template = PdfTemplateSample(
        source_pdf="sample.pdf",
        inferred_sections=["Executive Summary", "Recommendations"],
        parser_backend="mineru-transformers",
    )

    parsed = ReportFormatCrew(settings=settings).run("- Findings by Theme\n- Source Appendix", pdf_template)

    assert parsed.sections == ["Findings by Theme", "Source Appendix", "Executive Summary", "Recommendations"]


def test_pdf_template_reader_uses_parser_then_llm_refinement_in_live_mode(monkeypatch) -> None:
    temp_root = Path("F:/Code/agent/output/test-temp")
    temp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(mkdtemp(prefix="test-live-pdf-template-", dir=temp_root))
    pdf_path = tmp_path / "sample-template.pdf"
    settings = Settings(
        project_root=tmp_path,
        output_root=tmp_path / "output",
        USE_LIVE_CREWS=True,
        RESEARCH_REPORT_MODEL="ollama/gemma4:31b-cloud",
        RESEARCH_REPORT_OPENROUTER_BASE_URL="http://localhost:11434/v1",
    )
    parser = PdfTemplateParser(
        settings=settings,
        parse_impl=lambda _pdf_path, _max_pages: PdfTemplateSample(
            source_pdf=str(pdf_path),
            extracted_markdown="Executive Summary\nFindings by Theme",
            inferred_sections=["Executive Summary", "Findings by Theme"],
            page_count=1,
            parser_backend="mineru-transformers",
        ),
    )
    monkeypatch.setattr(
        pdf_template_crew_module,
        "kickoff_with_retry",
        lambda crew, inputs, settings: SimpleNamespace(
            json_dict={
                "source_pdf": str(pdf_path),
                "extracted_markdown": "## Executive Summary\n\n## Findings by Theme",
                "inferred_sections": ["Executive Summary", "Findings by Theme"],
                "page_count": 1,
                "parser_backend": "mineru-transformers",
            }
        ),
    )

    parsed = PdfTemplateCrew(settings=settings, parser=parser).run(str(pdf_path))

    assert parsed.inferred_sections == ["Executive Summary", "Findings by Theme"]
    assert parsed.extracted_markdown.startswith("## Executive Summary")
