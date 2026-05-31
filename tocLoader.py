"""
PDF table of contents loader.

Uses PDF outline/bookmarks when available (PyMuPDF), otherwise scans early
pages for textual contents lines (pypdf).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

DEFAULT_MAX_TEXT_PAGES = 15
DEFAULT_CONTENTS_KEYWORDS = ("table of contents", "contents", "chapter", "section")


class TocEntry(TypedDict):
    level: int
    title: str
    page: int | None
    source: str


def format_toc_entries(entries: list[TocEntry]) -> str:
    """Format TOC entries as indented plain text."""
    lines = []
    for entry in entries:
        indent = "  " * max(entry["level"] - 1, 0)
        page = entry["page"]
        page_label = f" (p. {page})" if page is not None else ""
        lines.append(f"{indent}{entry['title']}{page_label}")
    return "\n".join(lines)


class TocLoader:
    """Load PDF table of contents via outline or textual fallback."""

    def __init__(
        self,
        *,
        max_text_pages: int = DEFAULT_MAX_TEXT_PAGES,
        contents_keywords: tuple[str, ...] = DEFAULT_CONTENTS_KEYWORDS,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.max_text_pages = max_text_pages
        self.contents_keywords = contents_keywords
        self._log = log_fn or (lambda msg: print(msg, flush=True))

    def load_from_outline(self, file_path: Path) -> list[TocEntry]:
        """Load TOC from PDF bookmark/outline tree (PyMuPDF)."""
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("Install pymupdf: pip install pymupdf") from exc

        doc = fitz.open(file_path)
        try:
            raw_toc = doc.get_toc()
        finally:
            doc.close()

        entries: list[TocEntry] = []
        for level, title, page in raw_toc:
            title = title.strip()
            if title:
                entries.append(
                    {
                        "level": int(level),
                        "title": title,
                        "page": int(page) if page else None,
                        "source": "outline",
                    }
                )
        return entries

    @staticmethod
    def parse_text_line(line: str) -> TocEntry | None:
        """Parse a single textual TOC line ending with a page number."""
        line = line.strip()
        if len(line) < 4:
            return None

        patterns = (
            re.compile(r"^(?P<title>.+?)\s*\.{2,}\s*(?P<page>\d{1,4})\s*$"),
            re.compile(r"^(?P<title>.+?)\s+(?P<page>\d{1,4})\s*$"),
            re.compile(
                r"^(?P<index>\d+(?:\.\d+)*)\s+(?P<title>.+?)\s+(?P<page>\d{1,4})\s*$"
            ),
        )
        for pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            groups = match.groupdict()
            title = groups.get("title", line).strip()
            page = int(groups["page"])
            level = 1
            if "index" in groups:
                level = groups["index"].count(".") + 1
            elif title.startswith(("Chapter ", "CHAPTER ", "Section ", "SECTION ")):
                level = 1
            return {"level": level, "title": title, "page": page, "source": "text"}
        return None

    def load_from_text(
        self,
        file_path: Path,
        max_pages: int | None = None,
    ) -> list[TocEntry]:
        """Load TOC by scanning early pages for contents-style lines (pypdf)."""
        from pypdf import PdfReader

        max_pages = max_pages or self.max_text_pages
        reader = PdfReader(str(file_path))
        page_limit = min(max_pages, len(reader.pages))
        entries: list[TocEntry] = []
        seen: set[tuple[str, int | None]] = set()

        for page_idx in range(page_limit):
            text = reader.pages[page_idx].extract_text() or ""
            in_contents_region = False

            for raw_line in text.splitlines():
                line = " ".join(raw_line.split())
                if not line:
                    continue

                lower = line.lower()
                if any(keyword in lower for keyword in self.contents_keywords):
                    in_contents_region = True

                parsed = self.parse_text_line(line)
                if parsed:
                    in_contents_region = True
                    key = (parsed["title"], parsed["page"])
                    if key not in seen:
                        seen.add(key)
                        entries.append(parsed)
                elif not in_contents_region:
                    continue

        return entries

    def load(self, file_path: Path | str) -> tuple[list[TocEntry], str]:
        """
        Load PDF table of contents using the best available method.

        Returns:
            (entries, method) where method is "outline", "text", or "none".
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path.resolve()}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")

        outline_entries = self.load_from_outline(path)
        if outline_entries:
            self._log(f"TOC loaded from PDF outline ({len(outline_entries)} entries).")
            return outline_entries, "outline"

        self._log("PDF outline empty; falling back to textual TOC scan...")
        text_entries = self.load_from_text(path)
        if text_entries:
            self._log(f"TOC loaded from text scan ({len(text_entries)} entries).")
            return text_entries, "text"

        self._log("No TOC entries found.")
        return [], "none"


def load_pdf_table_of_contents(
    file_path: Path | str,
    *,
    max_text_pages: int = DEFAULT_MAX_TEXT_PAGES,
    contents_keywords: tuple[str, ...] = DEFAULT_CONTENTS_KEYWORDS,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[TocEntry], str]:
    """Convenience wrapper around TocLoader.load()."""
    loader = TocLoader(
        max_text_pages=max_text_pages,
        contents_keywords=contents_keywords,
        log_fn=log_fn,
    )
    return loader.load(file_path)
