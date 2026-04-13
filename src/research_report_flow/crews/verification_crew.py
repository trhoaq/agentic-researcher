from __future__ import annotations

from ..config import Settings, get_settings
from ..models import CandidateSource, FetchedSource, VerifiedSource
from ..tools import SourceFetcher, normalize_url


class VerificationCrew:
    def __init__(self, settings: Settings | None = None, fetcher: SourceFetcher | None = None) -> None:
        self.settings = settings or get_settings()
        self.fetcher = fetcher or SourceFetcher(settings=self.settings)

    def run(self, candidates: list[CandidateSource]) -> list[VerifiedSource]:
        seen: set[str] = set()
        verified: list[VerifiedSource] = []

        for source in candidates:
            normalized = normalize_url(str(source.url))
            if normalized in seen:
                continue
            seen.add(normalized)

            notes: list[str] = []
            fetched: FetchedSource | None = None
            try:
                fetched = self.fetcher.fetch(str(source.url))
                notes.append(f"reachable:{fetched.status_code}")
            except Exception as exc:
                notes.append(f"fetch_failed:{exc}")
                continue

            title = source.title or (fetched.title if fetched else "Untitled Source")
            verified.append(
                VerifiedSource(
                    source_id=source.source_id,
                    title=title,
                    url=str(source.url),
                    normalized_url=normalized,
                    source_type=source.source_type,
                    query=source.query,
                    snippet=source.snippet or (fetched.excerpt if fetched else ""),
                    authors=source.authors,
                    venue=source.venue,
                    year=source.year,
                    citation_label=f"[{len(verified)+1}] {title}",
                    verification_notes=notes,
                )
            )
        return verified
