from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from crewai.flow import Flow, listen, start
from pydantic import PrivateAttr

from .config import Settings, get_settings
from .crews import (
    PlanningCrew,
    PdfTemplateCrew,
    RefinementCrew,
    ReportFormatCrew,
    ReportRequirementsCrew,
    ResearchCrew,
    ScholarResearchCrew,
    SynthesisCrew,
    VerificationCrew,
    WritingCrew,
)
from .models import CandidateSource
from .state import ResearchReportState
from .tools import SourceRegistry


class ResearchReportFlow(Flow[ResearchReportState]):
    _settings: Settings = PrivateAttr()
    _requirements_crew: ReportRequirementsCrew = PrivateAttr()
    _pdf_template_crew: PdfTemplateCrew = PrivateAttr()
    _format_crew: ReportFormatCrew = PrivateAttr()
    _planning_crew: PlanningCrew = PrivateAttr()
    _research_crew: ResearchCrew = PrivateAttr()
    _scholar_research_crew: ScholarResearchCrew = PrivateAttr()
    _verification_crew: VerificationCrew = PrivateAttr()
    _synthesis_crew: SynthesisCrew = PrivateAttr()
    _writing_crew: WritingCrew = PrivateAttr()
    _refinement_crew: RefinementCrew = PrivateAttr()

    def __init__(
        self,
        settings: Settings | None = None,
        requirements_crew: ReportRequirementsCrew | None = None,
        pdf_template_crew: PdfTemplateCrew | None = None,
        format_crew: ReportFormatCrew | None = None,
        planning_crew: PlanningCrew | None = None,
        research_crew: ResearchCrew | None = None,
        scholar_research_crew: ScholarResearchCrew | None = None,
        verification_crew: VerificationCrew | None = None,
        synthesis_crew: SynthesisCrew | None = None,
        writing_crew: WritingCrew | None = None,
        refinement_crew: RefinementCrew | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self._requirements_crew = requirements_crew or ReportRequirementsCrew(self._settings)
        self._pdf_template_crew = pdf_template_crew or PdfTemplateCrew(self._settings)
        self._format_crew = format_crew or ReportFormatCrew(self._settings)
        self._planning_crew = planning_crew or PlanningCrew(self._settings)
        self._research_crew = research_crew or ResearchCrew(self._settings)
        self._scholar_research_crew = scholar_research_crew or ScholarResearchCrew(self._settings)
        self._verification_crew = verification_crew or VerificationCrew(self._settings)
        self._synthesis_crew = synthesis_crew or SynthesisCrew(self._settings)
        self._writing_crew = writing_crew or WritingCrew(self._settings)
        self._refinement_crew = refinement_crew or RefinementCrew(self._settings)

    @start()
    def initialize(
        self,
        brief: str = "",
        format_instructions: str = "",
        format_pdf_path: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.state.brief = brief or self.state.brief
        self.state.format_instructions = format_instructions or self.state.format_instructions
        self.state.format_pdf_path = format_pdf_path or self.state.format_pdf_path
        resolved_output_dir = output_dir or self.state.output_dir
        target_dir = Path(resolved_output_dir) if resolved_output_dir else self._settings.output_root / self.state.run_name
        target_dir.mkdir(parents=True, exist_ok=True)
        self.state.output_dir = str(target_dir)

    @listen(initialize)
    def parse_requirements(self) -> None:
        self.state.requirements = self._requirements_crew.run(self.state.brief)
        self._write_artifact("requirements.json", self.state.requirements.model_dump(mode="json"))

    @listen(parse_requirements)
    def parse_pdf_template(self) -> None:
        if not self.state.format_pdf_path:
            self.state.pdf_template = None
            return
        self.state.pdf_template = self._pdf_template_crew.run(self.state.format_pdf_path)
        self._write_artifact("pdf-template.json", self.state.pdf_template.model_dump(mode="json"))

    @listen(parse_pdf_template)
    def parse_format(self) -> None:
        self.state.format_spec = self._format_crew.run(self.state.format_instructions, self.state.pdf_template)
        self._write_artifact("format-spec.json", self.state.format_spec.model_dump(mode="json"))

    @listen(parse_format)
    def plan(self) -> None:
        self.state.research_plan, self.state.outline = self._planning_crew.run(self.state.requirements, self.state.format_spec)
        if self._settings.max_agenda_items > 0:
            self.state.research_plan.search_agenda = self.state.research_plan.search_agenda[: self._settings.max_agenda_items]
            self.state.research_plan.key_questions = self.state.research_plan.key_questions[: self._settings.max_agenda_items]
        self._write_artifact("research-plan.json", self.state.research_plan.model_dump(mode="json"))
        self._write_artifact("outline.json", self.state.outline.model_dump(mode="json"))

    @listen(plan)
    def dual_research(self) -> None:
        assert self.state.research_plan is not None
        if self._settings.dual_research_parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                web_future = pool.submit(self._research_crew.run, self.state.research_plan.search_agenda)
                scholar_future = pool.submit(self._scholar_research_crew.run, self.state.research_plan.search_agenda)
                self.state.web_candidates = web_future.result()
                self.state.scholar_candidates = scholar_future.result()
        else:
            self.state.web_candidates = self._research_crew.run(self.state.research_plan.search_agenda)
            self.state.scholar_candidates = self._scholar_research_crew.run(self.state.research_plan.search_agenda)
        self._write_artifact("web-candidates.json", [item.model_dump(mode="json") for item in self.state.web_candidates])
        self._write_artifact("scholar-candidates.json", [item.model_dump(mode="json") for item in self.state.scholar_candidates])

    @listen(dual_research)
    def merge_candidates(self) -> None:
        merged = self.state.web_candidates + self.state.scholar_candidates
        seen: set[tuple[str, str]] = set()
        deduped: list[CandidateSource] = []
        for source in merged:
            key = (source.source_type, str(source.url))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        self.state.merged_candidates = deduped
        self._write_artifact("merged-candidates.json", [item.model_dump(mode="json") for item in deduped])

    @listen(merge_candidates)
    def verify_sources(self) -> None:
        self.state.verified_sources = self._verification_crew.run(self.state.merged_candidates)
        registry = SourceRegistry(Path(self.state.output_dir) / "verified-sources.json")
        registry.save(self.state.verified_sources)
        self.state.artifact_paths["verified_sources"] = str(registry.path)

    @listen(verify_sources)
    def synthesize(self) -> None:
        self.state.synthesis = self._synthesis_crew.run(self.state.outline, self.state.verified_sources)
        self._write_artifact("synthesis.json", self.state.synthesis.model_dump(mode="json"))

    @listen(synthesize)
    def draft(self) -> None:
        self.state.draft_report = self._writing_crew.run(
            self.state.requirements,
            self.state.format_spec,
            self.state.outline,
            self.state.synthesis,
            self.state.verified_sources,
        )
        self._write_artifact("draft.md", self.state.draft_report.markdown, raw_text=True)

    @listen(draft)
    def refine(self) -> None:
        self.state.refinement = self._refinement_crew.run(
            self.state.requirements,
            self.state.format_spec,
            self.state.synthesis,
            self.state.draft_report,
        )
        self._write_artifact("final-report.md", self.state.refinement.markdown, raw_text=True)
        self._write_artifact("refinement.json", self.state.refinement.model_dump(mode="json"))
        self.state.final_report_path = str(Path(self.state.output_dir) / "final-report.md")

    def _write_artifact(self, filename: str, payload: object, raw_text: bool = False) -> None:
        path = Path(self.state.output_dir) / filename
        if raw_text:
            path.write_text(str(payload), encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.state.artifact_paths[filename] = str(path)
