from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import ReportRequirements
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class ReportRequirementsCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, brief: str) -> ReportRequirements:
        if not self.settings.should_use_live_crews():
            return self._fallback_parse(brief)

        try:
            agent = Agent(
                role="Requirements Analyst",
                goal="Turn the raw brief into a structured report requirement contract.",
                backstory="You normalize user research intents into explicit, testable report requirements.",
                llm=self.settings.llm_model,
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Read the brief and extract topic, audience, objectives, depth, constraints, "
                    "must-answer questions, and missing information.\n\nBrief:\n{brief}"
                ),
                expected_output="A structured ReportRequirements object.",
                output_pydantic=ReportRequirements,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={"brief": brief},
                settings=self.settings,
            )
            return parse_structured_output(result, ReportRequirements)
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="requirements")
                return self._fallback_parse(brief)
            raise

    def _fallback_parse(self, brief: str) -> ReportRequirements:
        lines = [line.strip(" -\t") for line in brief.splitlines() if line.strip()]
        topic = lines[0].lstrip("# ").strip() if lines else "Untitled Research Topic"
        objectives = [line for line in lines if line.lower().startswith(("goal:", "objective:", "objectives:"))]
        questions = [line for line in lines if "?" in line]
        constraints = [line for line in lines if line.lower().startswith(("constraint:", "constraints:", "must:"))]
        return ReportRequirements(
            topic=topic,
            objectives=[item.split(":", 1)[-1].strip() for item in objectives] or [topic],
            constraints=[item.split(":", 1)[-1].strip() for item in constraints],
            must_answer_questions=questions,
            missing_information=[],
        )
