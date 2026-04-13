from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..models import VerifiedSource


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=False)))
    normalized_path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, query, ""))


class SourceRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, sources: list[VerifiedSource]) -> None:
        payload = [source.model_dump(mode="json") for source in sources]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> list[VerifiedSource]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [VerifiedSource.model_validate(item) for item in data]
