from __future__ import annotations

from ..config import Settings, get_settings
from ..models import PdfTemplateSample, ReportFormatSpec


class ReportFormatCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, format_instructions: str, pdf_template: PdfTemplateSample | None = None) -> ReportFormatSpec:
        return self._fallback_parse(format_instructions, pdf_template)

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
