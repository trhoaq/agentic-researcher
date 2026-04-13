from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..config import Settings, get_settings
from ..models import ReportFormatSpec, ReportOutline, ReportOutlineSection, ReportRequirements, ResearchPlan, SearchAgendaItem
from .base import kickoff_with_retry, log_degrade, parse_structured_output, should_degrade_to_fallback


class PlanningOutput(ResearchPlan):
    outline: ReportOutline


class PlanningCrew:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, requirements: ReportRequirements, format_spec: ReportFormatSpec) -> tuple[ResearchPlan, ReportOutline]:
        if not self.settings.should_use_live_crews():
            plan = self._fallback_plan(requirements, format_spec)
            return ResearchPlan(key_questions=plan.key_questions, search_agenda=plan.search_agenda), plan.outline

        try:
            agent = Agent(
                role="Research Planner",
                goal="Create a research question set, search agenda, and report outline.",
                backstory="You break research work into explicit questions and section-level coverage.",
                llm=self.settings.llm_model,
                verbose=self.settings.verbose_crews,
            )
            task = Task(
                description=(
                    "Using the normalized requirements and report format, create a research plan with key questions, "
                    "search agenda, and an outline aligned to the report format.\n\nRequirements:\n{requirements}\n\nFormat:\n{format_spec}"
                ),
                expected_output="A structured planning payload with search agenda and outline.",
                output_pydantic=PlanningOutput,
                agent=agent,
            )
            result = kickoff_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self.settings.verbose_crews),
                inputs={
                    "requirements": requirements.model_dump_json(indent=2),
                    "format_spec": format_spec.model_dump_json(indent=2),
                },
                settings=self.settings,
            )
            parsed = parse_structured_output(result, PlanningOutput)
            return ResearchPlan(key_questions=parsed.key_questions, search_agenda=parsed.search_agenda), parsed.outline
        except Exception as exc:
            if self.settings.allow_live_fallback and should_degrade_to_fallback(exc):
                log_degrade(exc, stage="planning")
                plan = self._fallback_plan(requirements, format_spec)
                return ResearchPlan(key_questions=plan.key_questions, search_agenda=plan.search_agenda), plan.outline
            raise

    def _fallback_plan(self, requirements: ReportRequirements, format_spec: ReportFormatSpec) -> PlanningOutput:
        questions = requirements.must_answer_questions or [
            f"What are the most important findings about {requirements.topic}?",
            f"What risks or unknowns should the report highlight for {requirements.topic}?",
        ]
        agenda = [
            SearchAgendaItem(
                question=question,
                keywords=[requirements.topic, *requirements.objectives[:2]],
            )
            for question in questions
        ]
        outline = ReportOutline(
            sections=[
                ReportOutlineSection(
                    title=section,
                    objective=f"Address the report section '{section}' for {requirements.topic}.",
                    questions_covered=questions,
                )
                for section in (format_spec.sections or self.settings.default_sections)
            ]
        )
        return PlanningOutput(key_questions=questions, search_agenda=agenda, outline=outline)
