from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import PdfTemplateSample
from ..tools import PdfTemplateParser, PdfTemplateParserTool
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class PdfTemplateCrew:
    def __init__(
        self,
        settings: Settings | None = None,
        parser: PdfTemplateParser | None = None,
        tool: PdfTemplateParserTool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.parser = parser or PdfTemplateParser(settings=self.settings)
        self.tool = tool or PdfTemplateParserTool(parser=self.parser)

    def run(self, pdf_path: str) -> PdfTemplateSample:
        if not self.settings.should_use_live_crews():
            return self.parser.parse(pdf_path)

        try:
            agent = Agent(
                role="PDF Template Reader",
                goal="Extract section structure and layout signals from a sample report PDF.",
                backstory="You analyze PDF templates so the format lane can imitate the provided report style.",
                llm=self.settings.llm_model,
                tools=[self.tool],
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Use the pdf_template_parser tool to parse the sample report PDF and return a structured PdfTemplateSample.\n\n"
                    "PDF path:\n{pdf_path}"
                ),
                expected_output="A structured PdfTemplateSample object.",
                output_pydantic=PdfTemplateSample,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={"pdf_path": pdf_path},
                settings=self.settings,
            )
            return parse_structured_output(result, PdfTemplateSample)
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="pdf_template")
                return self.parser.parse(pdf_path)
            raise
