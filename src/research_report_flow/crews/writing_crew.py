from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import DraftReport, EvidenceSynthesis, ReportFormatSpec, ReportOutline, ReportRequirements, VerifiedSource
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class WritingCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        requirements: ReportRequirements,
        format_spec: ReportFormatSpec,
        outline: ReportOutline,
        synthesis: EvidenceSynthesis,
        sources: list[VerifiedSource],
    ) -> DraftReport:
        if not self.settings.should_use_live_crews():
            return self._fallback_write(requirements, format_spec, synthesis, sources)

        try:
            agent = Agent(
                role="Report Writer",
                goal="Draft a markdown report from structured evidence and the requested format.",
                backstory="You write only from verified evidence and honor the user’s requested structure.",
                llm=self.settings.llm_model,
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Draft the final markdown report using only the synthesis and verified sources.\n\n"
                    "Requirements:\n{requirements_json}\n\nFormat:\n{format_json}\n\nOutline:\n{outline_json}\n\n"
                    "Synthesis:\n{synthesis_json}\n\nSources:\n{sources_json}"
                ),
                expected_output="A structured DraftReport object containing markdown.",
                output_pydantic=DraftReport,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={
                    "requirements_json": requirements.model_dump_json(indent=2),
                    "format_json": format_spec.model_dump_json(indent=2),
                    "outline_json": outline.model_dump_json(indent=2),
                    "synthesis_json": synthesis.model_dump_json(indent=2),
                    "sources_json": "[" + ", ".join(source.model_dump_json() for source in sources) + "]",
                },
                settings=self.settings,
            )
            return parse_structured_output(result, DraftReport)
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="writing")
                return self._fallback_write(requirements, format_spec, synthesis, sources)
            raise

    def _fallback_write(
        self,
        requirements: ReportRequirements,
        format_spec: ReportFormatSpec,
        synthesis: EvidenceSynthesis,
        sources: list[VerifiedSource],
    ) -> DraftReport:
        lines = [f"# {format_spec.title or requirements.topic}", ""]

        for section in format_spec.sections or self.settings.default_sections:
            lines.append(f"## {section}")
            if section == "Executive Summary":
                lines.append(f"This report reviews {requirements.topic} for a {requirements.audience} audience.")
            elif section == "Research Objective":
                lines.extend(f"- {objective}" for objective in (requirements.objectives or [requirements.topic]))
            elif section == "Key Questions":
                items = requirements.must_answer_questions or synthesis.unknowns or ["No explicit questions supplied."]
                lines.extend(f"- {question}" for question in items)
            elif section == "Findings by Theme":
                for finding in synthesis.findings:
                    lines.append(f"### {finding.theme}")
                    lines.append(finding.summary)
            elif section == "Evidence Table":
                lines.append("| Claim | Sources | Confidence |")
                lines.append("| --- | --- | --- |")
                for row in synthesis.evidence_rows:
                    lines.append(f"| {row.claim} | {', '.join(row.source_ids)} | {row.confidence} |")
            elif section == "Risks / Conflicts / Unknowns":
                items = synthesis.unknowns or ["No material unknowns captured during this run."]
                lines.extend(f"- {item}" for item in items)
            elif section == "Recommendations":
                lines.append(f"- Prioritize follow-up on {requirements.topic} with the strongest verified sources first.")
                lines.append("- Review the open-web and scholar evidence together before making irreversible decisions.")
            elif section == "Source Appendix":
                for source in sources:
                    lines.append(f"- {source.citation_label}: {source.url}")
            else:
                lines.append(f"Section '{section}' is reserved by the requested format.")
            lines.append("")

        return DraftReport(markdown="\n".join(lines).strip() + "\n")
