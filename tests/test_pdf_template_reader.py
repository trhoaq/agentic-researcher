from __future__ import annotations

from pathlib import Path

from research_report_flow.config import Settings
from research_report_flow.crews.format_crew import ReportFormatCrew
from research_report_flow.crews.pdf_template_crew import PdfTemplateCrew
from research_report_flow.models import PdfTemplateSample
from research_report_flow.tools import PdfTemplateParser


def test_pdf_template_reader_infers_sections_and_feeds_format() -> None:
    tmp_path = Path("F:/Code/agent/.tmp/test-pdf-template")
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_path / "sample-template.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    settings = Settings(project_root=tmp_path, output_root=tmp_path / "output", use_live_crews=False)
    parser = PdfTemplateParser(
        settings=settings,
        parse_impl=lambda _pdf_path, _max_pages: PdfTemplateSample(
            source_pdf=str(pdf_path),
            extracted_markdown="Executive Summary\nFindings by Theme\nSource Appendix",
            inferred_sections=["Executive Summary", "Findings by Theme", "Source Appendix"],
            page_count=1,
            parser_backend="mineru-transformers",
        ),
    )
    pdf_template = PdfTemplateCrew(settings=settings, parser=parser).run(str(pdf_path))
    format_spec = ReportFormatCrew(settings=settings).run("", pdf_template)

    assert pdf_template.parser_backend == "mineru-transformers"
    assert "Executive Summary" in pdf_template.inferred_sections
    assert "Source Appendix" in format_spec.sections
