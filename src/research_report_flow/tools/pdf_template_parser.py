from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Callable

import fitz
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from ..config import Settings, get_settings
from ..models import PdfTemplateSample


class PdfTemplateParseArgs(BaseModel):
    pdf_path: str
    max_pages: int | None = None


class PdfTemplateParser:
    def __init__(
        self,
        settings: Settings | None = None,
        parse_impl: Callable[[str, int | None], PdfTemplateSample] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._parse_impl = parse_impl
        self._mineru_client = None

    def parse(self, pdf_path: str, max_pages: int | None = None) -> PdfTemplateSample:
        if self._parse_impl is not None:
            return self._parse_impl(pdf_path, max_pages)
        return self._parse_with_mineru(pdf_path, max_pages)

    def _parse_with_mineru(self, pdf_path: str, max_pages: int | None = None) -> PdfTemplateSample:
        try:
            from PIL import Image
            from mineru_vl_utils import MinerUClient
            from mineru_vl_utils.post_process import json2md
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(_mineru_setup_error_message(self.settings.pdf_template_model_id, exc)) from exc

        if self._mineru_client is None:
            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.settings.pdf_template_model_id,
                    dtype="auto",
                    device_map="auto",
                )
                processor = AutoProcessor.from_pretrained(
                    self.settings.pdf_template_model_id,
                    use_fast=True,
                )
                self._mineru_client = MinerUClient(
                    backend="transformers",
                    model=model,
                    processor=processor,
                    image_analysis=False,
                )
            except Exception as exc:
                raise RuntimeError(_mineru_setup_error_message(self.settings.pdf_template_model_id, exc)) from exc

        max_pages = max_pages or self.settings.pdf_template_max_pages
        markdown_chunks: list[str] = []
        doc = fitz.open(pdf_path)
        try:
            for page_index, page in enumerate(doc):
                if page_index >= max_pages:
                    break
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                content_list = self._mineru_client.two_step_extract(image)
                markdown_chunks.append(json2md(content_list))
        finally:
            doc.close()

        extracted_markdown = "\n\n".join(chunk.strip() for chunk in markdown_chunks if chunk.strip())
        return PdfTemplateSample(
            source_pdf=pdf_path,
            extracted_markdown=extracted_markdown,
            inferred_sections=_infer_sections(extracted_markdown, self.settings.default_sections),
            page_count=min(len(markdown_chunks), max_pages),
            parser_backend="mineru-transformers",
        )


class PdfTemplateParserTool(BaseTool):
    name: str = "pdf_template_parser"
    description: str = "Parse a PDF report template and infer sections and structure."
    args_schema: type[BaseModel] = PdfTemplateParseArgs
    _parser: PdfTemplateParser = PrivateAttr()

    def __init__(self, parser: PdfTemplateParser | None = None, **data: object) -> None:
        super().__init__(**data)
        self._parser = parser or PdfTemplateParser()

    def _run(self, pdf_path: str, max_pages: int | None = None) -> str:
        return json.dumps(
            self._parser.parse(pdf_path=pdf_path, max_pages=max_pages).model_dump(mode="json"),
            ensure_ascii=False,
        )


def _infer_sections(extracted_markdown: str, default_sections: list[str]) -> list[str]:
    known = {section.lower(): section for section in default_sections}
    found: list[str] = []

    for raw_line in extracted_markdown.splitlines():
        line = raw_line.strip().strip("#").strip()
        if not line or len(line) > 100:
            continue
        lowered = line.lower()
        if lowered in known and known[lowered] not in found:
            found.append(known[lowered])
            continue
        if re.fullmatch(r"[A-Z][A-Za-z0-9 /&:-]{2,80}", line) and line.count(".") <= 1:
            if line not in found:
                found.append(line)

    return found or default_sections


def _mineru_setup_error_message(model_id: str, exc: Exception) -> str:
    return (
        "MinerU PDF parser is required but unavailable. "
        f"Failed to load model '{model_id}'. "
        "Please install runtime dependencies and download/load the model first.\n"
        "Suggested setup:\n"
        "1) python -m pip install \"mineru-vl-utils[transformers]\" transformers torch\n"
        f"2) python -c \"from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; "
        f"AutoProcessor.from_pretrained('{model_id}'); "
        f"Qwen2VLForConditionalGeneration.from_pretrained('{model_id}', dtype='auto', device_map='auto')\"\n"
        f"Original error: {exc}"
    )
