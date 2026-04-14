from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_report_flow.crews.base import parse_structured_output
from research_report_flow.models import DraftReport, EvidenceSynthesis


def test_parse_structured_output_rejects_empty_text_with_value_error() -> None:
    with pytest.raises(ValueError, match="empty text"):
        parse_structured_output(SimpleNamespace(raw="   "), DraftReport)


def test_parse_structured_output_accepts_embedded_json_object() -> None:
    result = SimpleNamespace(raw='Result:\n{"markdown": "# Report"}')

    parsed = parse_structured_output(result, DraftReport)

    assert parsed.markdown == "# Report"


def test_parse_structured_output_accepts_fenced_json_object() -> None:
    result = SimpleNamespace(raw='```json\n{"markdown": "# Report"}\n```')

    parsed = parse_structured_output(result, DraftReport)

    assert parsed.markdown == "# Report"


def test_parse_structured_output_coerces_yaml_like_evidence_synthesis() -> None:
    result = SimpleNamespace(
        raw="""
evidence:
  - title: "Comparison of approaches"
    source: "src-1"
    confidence: high
    notes: "Useful tradeoff summary."
findings:
  - title: "Architecture"
    description: "Separate retrieval from synthesis."
    source: "src-1"
unknowns:
  - "Production latency depends on model choice."
"""
    )

    parsed = parse_structured_output(result, EvidenceSynthesis)

    assert parsed.evidence_rows[0].claim == "Comparison of approaches"
    assert parsed.evidence_rows[0].source_ids == ["src-1"]
    assert parsed.findings[0].theme == "Architecture"
    assert parsed.findings[0].evidence_source_ids == ["src-1"]
    assert parsed.unknowns == ["Production latency depends on model choice."]
