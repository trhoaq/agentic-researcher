from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class ReportRequirements(BaseModel):
    topic: str
    audience: str = "general"
    objectives: list[str] = Field(default_factory=list)
    depth: str = "standard"
    constraints: list[str] = Field(default_factory=list)
    must_answer_questions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class ReportFormatSpec(BaseModel):
    title: str = "Research Report"
    sections: list[str] = Field(default_factory=list)
    tone: str = "analytical"
    style: str = "markdown"
    citation_style: str = "inline links"
    output_rules: list[str] = Field(default_factory=list)


class PdfTemplateSample(BaseModel):
    source_pdf: str
    extracted_markdown: str = ""
    inferred_sections: list[str] = Field(default_factory=list)
    page_count: int = 0
    parser_backend: str = "mineru-transformers"


class SearchAgendaItem(BaseModel):
    question: str
    keywords: list[str] = Field(default_factory=list)
    recency_days: int | None = None
    scholar_year_from: int | None = None


class ReportOutlineSection(BaseModel):
    title: str
    objective: str
    questions_covered: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    key_questions: list[str] = Field(default_factory=list)
    search_agenda: list[SearchAgendaItem] = Field(default_factory=list)


class ReportOutline(BaseModel):
    sections: list[ReportOutlineSection] = Field(default_factory=list)


class CandidateSource(BaseModel):
    source_id: str
    title: str
    url: HttpUrl | str
    source_type: Literal["web", "scholar"]
    query: str
    snippet: str = ""
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None


class VerifiedSource(BaseModel):
    source_id: str
    title: str
    url: HttpUrl | str
    normalized_url: str
    source_type: Literal["web", "scholar"]
    query: str
    snippet: str = ""
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    citation_label: str
    verification_notes: list[str] = Field(default_factory=list)


class EvidenceRow(BaseModel):
    claim: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    notes: str = ""


class ThematicFinding(BaseModel):
    theme: str
    summary: str
    evidence_source_ids: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class EvidenceSynthesis(BaseModel):
    evidence_rows: list[EvidenceRow] = Field(default_factory=list)
    findings: list[ThematicFinding] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class DraftReport(BaseModel):
    markdown: str


class RefinementResult(BaseModel):
    markdown: str
    revision_notes: list[str] = Field(default_factory=list)


class FetchedSource(BaseModel):
    url: HttpUrl | str
    final_url: str
    status_code: int
    title: str | None = None
    excerpt: str = ""
