"""Fine-stage verbatim page-text extraction (advanced, default off).

Called unconditionally after fine render. ``extractor="none"`` returns immediately
with no extra model or parser calls. ``vlm`` / ``ocr`` / ``mineru`` fill page text
used to overwrite M0 band content when the advanced switch is on. When ``vlm``
raises on a page, that page falls back to ``ocr``; when ``mineru`` raises, the
scope falls back to ``ocr``; ``ocr`` errors fall back to ``none`` (keeping M0).

``extract_fine_page_details`` additionally collects LaTeX formulas:
``mineru`` reads them from layout.json (inline_formula spans and
interline_equation blocks, with page-pt bboxes); ``vlm`` parses ``$..$`` /
``$$..$$`` spans out of the transcription. ``none``/``ocr`` never yield
formulas — plain OCR is not trusted for math.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
import shutil
from typing import Any

from loguru import logger

from app.services.document_agent.tools.ocr_pages import ocr_rendered_pages
from app.services.document_parser.formula.extractor import extract_latex_formulas
from app.services.document_parser.formula.redelimit import (
    looks_mathematical,
    redelimit_transcription,
)
from app.services.document_parser.formula.models import FormulaRecord
from shared.services.ai.summary.engine import transcribe

_PAGE_RENDER_DPI = 144  # matches page_renderer.render_document_pages default
_PT_PER_INCH = 72.0


@dataclass
class FinePageText:
    """Fine-stage extraction result: page text plus extracted formulas."""

    text_by_page: dict[int, str] = field(default_factory=dict)
    formulas_by_page: dict[int, list[FormulaRecord]] = field(default_factory=dict)


def _build_image_path_by_page(
    rendered: list[Any],
    pages: list[int],
) -> dict[int, str]:
    page_set = set(pages)
    image_path_by_page: dict[int, str] = {}
    for item in rendered:
        raw_page = getattr(item, "page_index", None)
        if raw_page is None:
            continue
        page = int(raw_page)
        if page not in page_set:
            continue
        image_path = str(getattr(item, "image_path", "") or "").strip()
        if image_path:
            image_path_by_page[page] = image_path
    return image_path_by_page


def _page_size_pt_by_page(
    rendered: list[Any],
    pages: list[int],
) -> dict[int, tuple[float, float]]:
    page_set = set(pages)
    sizes: dict[int, tuple[float, float]] = {}
    for item in rendered:
        raw_page = getattr(item, "page_index", None)
        if raw_page is None:
            continue
        page = int(raw_page)
        if page not in page_set:
            continue
        width = getattr(item, "width", None)
        height = getattr(item, "height", None)
        if width and height:
            sizes[page] = (float(width), float(height))
    return sizes


def _extract_via_ocr(
    *,
    image_path_by_page: Mapping[int, str],
    pages: list[int],
) -> dict[int, str]:
    subset = {
        page: image_path_by_page[page]
        for page in pages
        if page in image_path_by_page
    }
    if not subset:
        return {}
    try:
        return ocr_rendered_pages(subset)
    except Exception as exc:
        logger.warning(
            "[page_text_extractor] ocr failed; keeping M0 text: {}",
            exc,
        )
        return {}


def extract_fine_page_text(
    *,
    pdf_path: str,
    rendered: list[Any],
    final_pages: list[int],
    extractor: str,
    vlm_model: str | None,
    output_dir: str,
    concurrency: int,
) -> dict[int, str]:
    """Return ``{page: verbatim text}`` for the requested extractor.

    ``none`` (default) returns ``{}`` so callers keep M0 band text. Unknown
    extractor names also return ``{}`` so the default-off path cannot spend.
    """
    return extract_fine_page_details(
        pdf_path=pdf_path,
        rendered=rendered,
        final_pages=final_pages,
        extractor=extractor,
        vlm_model=vlm_model,
        output_dir=output_dir,
        concurrency=concurrency,
    ).text_by_page


def extract_fine_page_details(
    *,
    pdf_path: str,
    rendered: list[Any],
    final_pages: list[int],
    extractor: str,
    vlm_model: str | None,
    output_dir: str,
    concurrency: int,
) -> FinePageText:
    """Return page text plus extracted formulas for the requested extractor."""
    resolved = (extractor or "none").strip().lower()
    pages = [int(page) for page in final_pages if int(page) > 0]
    if resolved == "none" or not pages:
        return FinePageText()
    image_path_by_page = _build_image_path_by_page(rendered, pages)
    if resolved == "ocr":
        return FinePageText(
            text_by_page=_extract_via_ocr(
                image_path_by_page=image_path_by_page,
                pages=pages,
            )
        )
    if resolved == "mineru":
        try:
            return _extract_via_mineru(
                pdf_path=pdf_path,
                final_pages=pages,
                output_dir=output_dir,
                page_size_pt_by_page=_page_size_pt_by_page(rendered, pages),
            )
        except Exception as exc:
            logger.warning(
                "[page_text_extractor] mineru failed ({}); falling back to ocr",
                exc,
            )
            return FinePageText(
                text_by_page=_extract_via_ocr(
                    image_path_by_page=image_path_by_page,
                    pages=pages,
                )
            )
    if resolved == "vlm":
        return _extract_via_vlm(
            image_path_by_page=image_path_by_page,
            final_pages=pages,
            vlm_model=vlm_model,
            concurrency=concurrency,
        )
    return FinePageText()


def _extract_via_vlm(
    *,
    image_path_by_page: Mapping[int, str],
    final_pages: list[int],
    vlm_model: str | None,
    concurrency: int,
) -> FinePageText:
    jobs = [
        (page, str(image_path_by_page.get(page) or ""))
        for page in final_pages
    ]
    if not jobs:
        return FinePageText()

    import gevent
    from gevent.pool import Pool as GeventPool

    def _one(page: int, image_path: str) -> tuple[int, str, Exception | None]:
        try:
            text = transcribe(
                image_paths=[image_path],
                model=vlm_model,
                usage_task="page_memory.fine_transcribe",
            )
            return page, (text or "").strip(), None
        except Exception as exc:
            return page, "", exc

    pool = GeventPool(size=min(max(1, concurrency), len(jobs)))
    greenlets = [pool.spawn(_one, page, image_path) for page, image_path in jobs]
    gevent.joinall(greenlets, raise_error=False)
    extracted: dict[int, str] = {}
    formulas: dict[int, list[FormulaRecord]] = {}
    failed_pages: list[int] = []
    for (job_page, _), greenlet in zip(jobs, greenlets, strict=True):
        error: BaseException | None
        text = ""
        if greenlet.value is None:
            error = getattr(greenlet, "exception", None) or RuntimeError(
                "vlm greenlet returned no result"
            )
        else:
            _, text, error = greenlet.value
        if error is not None:
            logger.warning(
                "[page_text_extractor] vlm transcribe failed on page {} ({}); falling back to ocr",
                job_page,
                error,
            )
            failed_pages.append(int(job_page))
            continue
        if text:
            page_formulas = extract_latex_formulas(text, page_num=int(job_page))
            if not page_formulas and looks_mathematical(text):
                # VLM transcriptions do not reliably follow the $..$
                # delimiter instruction (same flakiness as the image
                # parser); one cheap text rewrite restores them.
                rewritten = redelimit_transcription(
                    text,
                    usage_task="page_memory.formula_redelimit",
                )
                if rewritten:
                    re_extracted = extract_latex_formulas(
                        rewritten, page_num=int(job_page)
                    )
                    if re_extracted:
                        text = rewritten
                        page_formulas = re_extracted
            extracted[int(job_page)] = text
            if page_formulas:
                formulas[int(job_page)] = [
                    FormulaRecord(latex=item.latex, page_num=int(job_page))
                    for item in page_formulas
                ]
    if failed_pages:
        extracted.update(
            _extract_via_ocr(
                image_path_by_page=image_path_by_page,
                pages=failed_pages,
            )
        )
    return FinePageText(text_by_page=extracted, formulas_by_page=formulas)


def _extract_via_mineru(
    *,
    pdf_path: str,
    final_pages: list[int],
    output_dir: str,
    page_size_pt_by_page: Mapping[int, tuple[float, float]] | None = None,
) -> FinePageText:
    from app.services.document_parser.formats.pdf.shard_splitter import (
        MergedShard,
        split_pdf,
    )
    from app.services.document_parser.providers.mineru.pdf_service import parse_via_full

    ordered = sorted({int(page) for page in final_pages})
    if not ordered:
        return FinePageText()
    work_dir = os.path.join(output_dir, "mineru")
    os.makedirs(work_dir, exist_ok=True)
    try:
        page_start = ordered[0]
        page_end = ordered[-1]
        exclude_pages = set(range(page_start, page_end + 1)) - set(ordered)
        live_shards = split_pdf(
            pdf_path,
            [MergedShard(0, page_start=page_start, page_end=page_end)],
            work_dir,
            exclude_pages=exclude_pages or None,
        )
        if not live_shards:
            return FinePageText()
        live = live_shards[0]
        parse_via_full(
            live.pdf_path,
            os.path.basename(live.pdf_path) or "scope.pdf",
            work_dir,
        )
        layout_path = os.path.join(work_dir, "layout.json")
        try:  # diagnostic keep-copy until page-coverage/bbox gaps are settled
            import shutil as _shutil

            _shutil.copyfile(layout_path, "/tmp/mineru_last_layout.json")
        except OSError:
            pass
        return _page_text_from_layout(
            layout_path,
            live.page_numbers,
            page_size_pt_by_page=page_size_pt_by_page,
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _page_text_from_layout(
    layout_json_path: str,
    page_numbers: list[int],
    page_size_pt_by_page: Mapping[int, tuple[float, float]] | None = None,
) -> FinePageText:
    if not os.path.exists(layout_json_path):
        raise FileNotFoundError(f"MinerU layout.json missing at {layout_json_path}")
    with open(layout_json_path, encoding="utf-8") as handle:
        layout = json.load(handle)
    if not isinstance(layout, dict):
        raise ValueError("MinerU layout.json root is not a JSON object")
    extracted: dict[int, str] = {}
    formulas: dict[int, list[FormulaRecord]] = {}
    for page in layout.get("pdf_info") or []:
        if not isinstance(page, dict):
            continue
        try:
            page_idx = int(page.get("page_idx", 0))
        except (TypeError, ValueError):
            continue
        if not (0 <= page_idx < len(page_numbers)):
            continue
        real_page = int(page_numbers[page_idx])
        size_pt = (page_size_pt_by_page or {}).get(real_page)
        page_formulas: list[FormulaRecord] = []
        texts = _block_texts(
            page.get("para_blocks") or [],
            page_num=real_page,
            page_size_pt=size_pt,
            formulas_out=page_formulas,
        )
        joined = "\n".join(part for part in texts if part)
        if joined:
            extracted[real_page] = joined
        if page_formulas:
            formulas[real_page] = page_formulas
        logger.info(
            "[fine_mineru] page {} -> text={} formulas={} size_pt={}",
            real_page,
            bool(joined),
            len(page_formulas),
            size_pt is not None,
        )
    return FinePageText(text_by_page=extracted, formulas_by_page=formulas)


def _block_texts(
    blocks: list[Any],
    *,
    page_num: int | None = None,
    page_size_pt: tuple[float, float] | None = None,
    formulas_out: list[FormulaRecord] | None = None,
) -> list[str]:
    texts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") == "interline_equation":
            latex = _interline_latex(block)
            if latex:
                texts.append(f"$${latex}$$")
                if formulas_out is not None and page_num is not None:
                    formulas_out.append(
                        FormulaRecord(
                            latex=latex,
                            page_num=page_num,
                            bbox_px=_bbox_pt_to_px(block.get("bbox"), page_size_pt),
                        )
                    )
                continue
        texts.extend(
            _block_texts(
                list(block.get("blocks") or []),
                page_num=page_num,
                page_size_pt=page_size_pt,
                formulas_out=formulas_out,
            )
        )
        for line in block.get("lines") or []:
            if not isinstance(line, dict):
                continue
            line_parts: list[str] = []
            for span in line.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                span_type = str(span.get("type") or "")
                content = str(span.get("content") or "").strip()
                if span_type in {"inline_formula", "interline_equation"} or span.get(
                    "latex"
                ):
                    latex = str(span.get("latex") or "").strip() or content
                    if latex:
                        line_parts.append(f"${latex}$")
                        if formulas_out is not None and page_num is not None:
                            formulas_out.append(
                                FormulaRecord(
                                    latex=latex,
                                    page_num=page_num,
                                    bbox_px=_bbox_pt_to_px(
                                        span.get("bbox"), page_size_pt
                                    ),
                                )
                            )
                        continue
                if content:
                    line_parts.append(content)
                    continue
                table_text = _table_html_text(span.get("html"))
                if table_text:
                    line_parts.append(table_text)
            if line_parts:
                texts.append(" ".join(line_parts))
    return texts


def _interline_latex(block: dict[str, Any]) -> str:
    """Resolve display-equation latex across MinerU layout.json variants."""
    latex = str(block.get("latex") or "").strip()
    if latex:
        return latex
    for line in block.get("lines") or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans") or []:
            if not isinstance(span, dict):
                continue
            candidate = str(span.get("latex") or "").strip() or str(
                span.get("content") or ""
            ).strip()
            if candidate:
                return candidate
    for nested in block.get("blocks") or []:
        if not isinstance(nested, dict):
            continue
        for line in nested.get("lines") or []:
            if not isinstance(line, dict):
                continue
            for span in line.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                candidate = str(span.get("content") or "").strip()
                if candidate:
                    return candidate
    return ""


def _bbox_pt_to_px(
    bbox: Any,
    page_size_pt: tuple[float, float] | None,
) -> list[int] | None:
    """Convert a layout.json bbox (PDF points, top-left) to dpi-144 pixels.

    Returns ``None`` on malformed input or when the page size is unknown so
    the formula falls back to whole-page verification.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or page_size_pt is None:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    scale = _PAGE_RENDER_DPI / _PT_PER_INCH
    width_px = page_size_pt[0] * scale
    height_px = page_size_pt[1] * scale
    return [
        max(0, min(int(x1 * scale), int(width_px))),
        max(0, min(int(y1 * scale), int(height_px))),
        max(0, min(int(x2 * scale), int(width_px))),
        max(0, min(int(y2 * scale), int(height_px))),
    ]


def _table_html_text(raw_html: Any) -> str:
    html = str(raw_html or "").strip()
    if not html:
        return ""
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(html, "html.parser")
    rows: list[str] = []
    for tr in soup.find_all("tr"):
        if not isinstance(tr, Tag):
            continue
        cells = [
            " ".join(cell.get_text(" ", strip=True).split())
            for cell in tr.find_all(["th", "td"])
        ]
        row_text = " | ".join(cell for cell in cells if cell)
        if row_text:
            rows.append(row_text)
    if rows:
        return "\n".join(rows)
    return " ".join(soup.get_text(" ", strip=True).split())
