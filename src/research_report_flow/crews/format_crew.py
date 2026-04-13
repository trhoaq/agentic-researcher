from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import PdfTemplateSample, ReportFormatSpec
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class ReportFormatCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, format_instructions: str, pdf_template: PdfTemplateSample | None = None) -> ReportFormatSpec:
        if not self.settings.should_use_live_crews():
            return self._fallback_parse(format_instructions, pdf_template)

        try:
            agent = Agent(
                role="Format Analyst",
                goal="Turn the user's requested report format into a concrete output contract.",
                backstory="You normalize report structure and style constraints into deterministic output rules.",
                llm=self.settings.llm_model,
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Read the report format instruction and extract title, sections, tone, style, citation style, "
                    "and output rules.\n\nFormat instruction:\n{format_instructions}\n\nPDF template analysis:\n{pdf_template_json}"
                ),
                expected_output="A structured ReportFormatSpec object.",
                output_pydantic=ReportFormatSpec,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={
                    "format_instructions": format_instructions,
                    "pdf_template_json": pdf_template.model_dump_json(indent=2) if pdf_template else "null",
                },
                settings=self.settings,
            )
            return parse_structured_output(result, ReportFormatSpec)
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="format")
                return self._fallback_parse(format_instructions, pdf_template)
            raise

    def _fallback_parse(self, format_instructions: str, pdf_template: PdfTemplateSample | None = None) -> ReportFormatSpec:
        sections = [
            line.strip(" -\t")
            for line in format_instructions.splitlines()
            if line.strip().startswith(("-", "*")) or line.strip().startswith(tuple(str(i) for i in range(1, 10)))
        ]
        normalized_sections = [section.split(".", 1)[-1].strip() if "." in section[:3] else section.lstrip("-* ").strip() for section in sections]
        merged_sections = normalized_sections or []
        if pdf_template:
            for section in pdf_template.inferred_sections:
                if section not in merged_sections:
                    merged_sections.append(section)
        return ReportFormatSpec(
            title="Research Report",
            sections=merged_sections or self.settings.default_sections,
            output_rules=[
                "Honor the requested section order.",
                *([f"Mirror headings inferred from sample PDF: {', '.join(pdf_template.inferred_sections)}"] if pdf_template else []),
            ],
        )
