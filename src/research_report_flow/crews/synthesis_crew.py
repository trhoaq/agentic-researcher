from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import EvidenceRow, EvidenceSynthesis, ReportOutline, ThematicFinding, VerifiedSource
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class SynthesisCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, outline: ReportOutline, verified_sources: list[VerifiedSource]) -> EvidenceSynthesis:
        if not self.settings.should_use_live_crews():
            return self._fallback_synthesize(outline, verified_sources)

        try:
            agent = Agent(
                role="Evidence Synthesizer",
                goal="Turn verified sources into evidence rows and thematic findings.",
                backstory="You cluster evidence, highlight contradictions, and preserve attribution.",
                llm=self.settings.llm_model,
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Use the verified sources to build an evidence table and thematic findings aligned to the report outline.\n\n"
                    "Outline:\n{outline_json}\n\nVerified sources:\n{sources_json}"
                ),
                expected_output="A structured EvidenceSynthesis object.",
                output_pydantic=EvidenceSynthesis,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={
                    "outline_json": outline.model_dump_json(indent=2),
                    "sources_json": "[" + ", ".join(source.model_dump_json() for source in verified_sources) + "]",
                },
                settings=self.settings,
            )
            return parse_structured_output(result, EvidenceSynthesis)
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="synthesis")
                return self._fallback_synthesize(outline, verified_sources)
            raise

    def _fallback_synthesize(self, outline: ReportOutline, verified_sources: list[VerifiedSource]) -> EvidenceSynthesis:
        evidence_rows = [
            EvidenceRow(
                claim=source.title,
                source_ids=[source.source_id],
                confidence="medium",
                notes=source.snippet[:180],
            )
            for source in verified_sources
        ]
        findings = [
            ThematicFinding(
                theme=section.title,
                summary=f"{section.title} is supported by {len(verified_sources)} verified sources in this run.",
                evidence_source_ids=[source.source_id for source in verified_sources[: min(3, len(verified_sources))]],
                open_questions=[],
            )
            for section in outline.sections
        ]
        unknowns = [] if verified_sources else ["No verified sources available."]
        return EvidenceSynthesis(evidence_rows=evidence_rows, findings=findings, unknowns=unknowns)
