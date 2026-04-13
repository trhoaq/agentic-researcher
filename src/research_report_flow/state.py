from __future__ import annotations

from datetime import datetime, UTC

from crewai.flow.flow import FlowState
from pydantic import Field

from .models import (
    CandidateSource,
    DraftReport,
    EvidenceSynthesis,
    PdfTemplateSample,
    RefinementResult,
    ReportFormatSpec,
    ReportOutline,
    ReportRequirements,
    ResearchPlan,
    VerifiedSource,
)


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class ResearchReportState(FlowState):
    brief: str = ""
    format_instructions: str = ""
    format_pdf_path: str | None = None
    run_name: str = Field(default_factory=utc_timestamp)
    output_dir: str = ""
    requirements: ReportRequirements | None = None
    pdf_template: PdfTemplateSample | None = None
    format_spec: ReportFormatSpec | None = None
    research_plan: ResearchPlan | None = None
    outline: ReportOutline | None = None
    web_candidates: list[CandidateSource] = Field(default_factory=list)
    scholar_candidates: list[CandidateSource] = Field(default_factory=list)
    merged_candidates: list[CandidateSource] = Field(default_factory=list)
    verified_sources: list[VerifiedSource] = Field(default_factory=list)
    synthesis: EvidenceSynthesis | None = None
    draft_report: DraftReport | None = None
    refinement: RefinementResult | None = None
    final_report_path: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
