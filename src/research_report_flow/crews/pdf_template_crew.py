from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import PdfTemplateSample
from ..tools import PdfTemplateParser
from .base import kickoff_with_retry, parse_structured_output


class PdfTemplateCrew:
    def __init__(
        self,
        settings: Settings | None = None,
        parser: PdfTemplateParser | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.parser = parser or PdfTemplateParser(settings=self.settings)

    def run(self, pdf_path: str) -> PdfTemplateSample:
        parsed_sample = self.parser.parse(pdf_path)
        if not self.settings.should_use_live_crews():
            return parsed_sample

        agent = Agent(
            role="PDF Template Reader",
            goal="Refine parsed PDF template output into a cleaner structure for downstream format analysis.",
            backstory="You improve parser output, preserve section structure, and clean noisy markdown without inventing content.",
            llm=self.settings.llm_model,
            verbose=self.settings.verbose_crews,
        )
        task = Task(
            description=(
                "You are given raw parser output from a sample report PDF. Clean the extracted markdown, preserve valid "
                "section headings, and return a refined PdfTemplateSample.\n\nReturn JSON only.\n\n"
                "PDF path:\n{pdf_path}\n\nParsed sample:\n{parsed_sample_json}"
            ),
            expected_output="A structured PdfTemplateSample object.",
            output_pydantic=PdfTemplateSample,
            agent=agent,
            markdown=False,
        )
        result = kickoff_with_retry(
            Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
            inputs={"pdf_path": pdf_path, "parsed_sample_json": parsed_sample.model_dump_json(indent=2)},
            settings=self.settings,
        )
        return parse_structured_output(result, PdfTemplateSample)
