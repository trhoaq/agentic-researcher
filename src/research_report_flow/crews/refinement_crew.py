from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import DraftReport, EvidenceSynthesis, RefinementResult, ReportFormatSpec, ReportRequirements
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class RefinementCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        requirements: ReportRequirements,
        format_spec: ReportFormatSpec,
        synthesis: EvidenceSynthesis,
        draft: DraftReport,
    ) -> RefinementResult:
        if not self.settings.should_use_live_crews():
            return self._fallback_refine(format_spec, synthesis, draft)

        try:
            agent = Agent(
                role="Report Critic",
                goal="Refine the draft for unsupported claims, gaps, and format mismatches.",
                backstory="You act as the final quality gate before export.",
                llm=self.settings.llm_model,
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Review the draft against the requirements, format contract, and synthesis. Return the final markdown "
                    "plus revision notes.\n\nRequirements:\n{requirements_json}\n\nFormat:\n{format_json}\n\n"
                    "Synthesis:\n{synthesis_json}\n\nDraft:\n{draft_markdown}"
                ),
                expected_output="A structured RefinementResult object.",
                output_pydantic=RefinementResult,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={
                    "requirements_json": requirements.model_dump_json(indent=2),
                    "format_json": format_spec.model_dump_json(indent=2),
                    "synthesis_json": synthesis.model_dump_json(indent=2),
                    "draft_markdown": draft.markdown,
                },
                settings=self.settings,
            )
            return parse_structured_output(result, RefinementResult)
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="refinement")
                return self._fallback_refine(format_spec, synthesis, draft)
            raise

    def _fallback_refine(
        self,
        format_spec: ReportFormatSpec,
        synthesis: EvidenceSynthesis,
        draft: DraftReport,
    ) -> RefinementResult:
        markdown = draft.markdown
        notes: list[str] = []
        for section in format_spec.sections or self.settings.default_sections:
            if f"## {section}" not in markdown:
                markdown += f"\n## {section}\nSection inserted during refinement to satisfy the format contract.\n"
                notes.append(f"Inserted missing section: {section}")
        if synthesis.unknowns and "Risks / Conflicts / Unknowns" not in markdown:
            markdown += "\n## Risks / Conflicts / Unknowns\n" + "\n".join(f"- {item}" for item in synthesis.unknowns) + "\n"
            notes.append("Added unknowns section from synthesis.")
        return RefinementResult(markdown=markdown, revision_notes=notes)
