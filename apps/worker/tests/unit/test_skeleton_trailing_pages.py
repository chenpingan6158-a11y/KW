"""Unit tests for trailing-page absorption in skeleton extraction."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("TMP_PATH", "/tmp/knowhere-test")
os.environ.setdefault("S3_BUCKET_NAME", "test-uploads")
os.environ.setdefault("S3_ACCESS_KEY_ID", "test")
os.environ.setdefault("S3_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("S3_TEMP_PATH", "/tmp")

from typing import Any


def _skel(path: str, start: int, end: int, *, level: int = 1,
          evidence: dict[str, Any] | None = None) -> "Any":
    from app.services.page_memory.skeleton_extractor import SectionSkeleton

    parts = path.split("/")
    return SectionSkeleton(
        section_path=path,
        level=level,
        start_page=start,
        end_page=end,
        title=parts[-1],
        parent_path="/".join(parts[:-1]) or None,
        evidence=evidence or {"resolver": "hierarchy_locator"},
    )


def test_last_chapter_absorbs_unlisted_trailing_pages() -> None:
    from app.services.page_memory.skeleton_extractor import (
        _extend_last_chapter_to_trailing_pages,
    )

    skeletons = [
        _skel("doc.pdf/Root", 1, 0, level=0, evidence={"source": "document_prefix"}),
        _skel("doc.pdf/Ch1", 1, 3),
        _skel("doc.pdf/Ch2", 4, 6),
    ]
    _extend_last_chapter_to_trailing_pages(
        skeletons, filename="doc.pdf", page_count=10, body_pages=list(range(1, 11))
    )
    by_title = {s.title: s for s in skeletons}
    assert by_title["Ch2"].end_page == 10
    assert by_title["Ch2"].evidence["trailing_pages_absorbed"] == 4
    assert by_title["Ch1"].end_page == 3  # untouched


def test_no_extension_when_toc_covers_everything() -> None:
    from app.services.page_memory.skeleton_extractor import (
        _extend_last_chapter_to_trailing_pages,
    )

    skeletons = [_skel("doc.pdf/Ch1", 1, 5), _skel("doc.pdf/Ch2", 6, 10)]
    _extend_last_chapter_to_trailing_pages(
        skeletons, filename="doc.pdf", page_count=10, body_pages=list(range(1, 11))
    )
    assert skeletons[1].end_page == 10
    assert "trailing_pages_absorbed" not in (skeletons[1].evidence or {})


def test_extension_respects_toc_pages_excluded_from_body() -> None:
    from app.services.page_memory.skeleton_extractor import (
        _extend_last_chapter_to_trailing_pages,
    )

    # Page 10 is a trailing TOC page; body ends at 9.
    skeletons = [_skel("doc.pdf/Ch1", 1, 6)]
    _extend_last_chapter_to_trailing_pages(
        skeletons, filename="doc.pdf", page_count=10, body_pages=list(range(1, 10))
    )
    assert skeletons[0].end_page == 9


def test_fallback_root_skeletons_untouched() -> None:
    from app.services.page_memory.skeleton_extractor import (
        _extend_last_chapter_to_trailing_pages,
    )

    skeletons = [
        _skel(
            "doc.pdf/Root",
            1,
            10,
            evidence={"source": "fallback_root", "reason": "no_toc"},
        )
    ]
    _extend_last_chapter_to_trailing_pages(
        skeletons, filename="doc.pdf", page_count=10, body_pages=list(range(1, 11))
    )
    assert skeletons[0].end_page == 10
    assert "trailing_pages_absorbed" not in (skeletons[0].evidence or {})


def test_toc_ownership_nodes_do_not_absorb() -> None:
    from app.services.page_memory.skeleton_extractor import (
        _extend_last_chapter_to_trailing_pages,
    )

    skeletons = [
        _skel("doc.pdf/Ch1", 1, 6),
        _skel("doc.pdf/Ch1/TOC", 2, 2, evidence={"source": "toc_node"}),
    ]
    _extend_last_chapter_to_trailing_pages(
        skeletons, filename="doc.pdf", page_count=9, body_pages=list(range(1, 10))
    )
    by_title = {s.title: s for s in skeletons}
    assert by_title["Ch1"].end_page == 9
    assert by_title["TOC"].end_page == 2
