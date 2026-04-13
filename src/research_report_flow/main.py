from __future__ import annotations

import argparse
from pathlib import Path

from .config import configure_runtime_environment, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CrewAI research report flow.")
    parser.add_argument("--brief-file", type=Path, help="Path to the report brief markdown/text file.")
    parser.add_argument("--format-file", type=Path, help="Path to the report format instruction file.")
    parser.add_argument("--format-pdf-file", type=Path, default=None, help="Optional PDF template file for the format lane.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for artifacts.")
    return parser


def main() -> None:
    configure_runtime_environment()
    from .flow import ResearchReportFlow

    parser = build_parser()
    args = parser.parse_args()

    if not args.brief_file:
        parser.error("--brief-file is required.")
    if not args.format_file and not args.format_pdf_file:
        parser.error("Provide at least one of --format-file or --format-pdf-file.")

    brief = args.brief_file.read_text(encoding="utf-8")
    format_instructions = args.format_file.read_text(encoding="utf-8") if args.format_file else ""
    flow = ResearchReportFlow(settings=get_settings())
    flow.kickoff(
        inputs={
            "brief": brief,
            "format_instructions": format_instructions,
            "format_pdf_path": str(args.format_pdf_file) if args.format_pdf_file else None,
            "output_dir": str(args.output_dir) if args.output_dir else None,
        }
    )
    print(flow.state.final_report_path)


if __name__ == "__main__":
    main()
