"""Render-back verification of extracted LaTeX formulas.

Each formula is rendered to a PNG (mathtext), judged against the original
source image by a VLM, and corrected at most once when the judge reads a
different formula off the source. The original transcription is never lost:
every failure path keeps a usable ``latex`` on the record.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, cast

from loguru import logger

from app.services.document_parser.formula.models import FormulaRecord
from app.services.document_parser.formula.renderer import render_formula_png
from shared.services.ai.prompt_service import build_prompt

_CROP_MARGIN_PX = 6


def formula_verify_enabled() -> bool:
    """Master switch for the render-back verification loop."""
    from shared.core.config import settings

    return bool(getattr(settings, "FORMULA_VERIFY_ENABLED", True))


def crop_bbox(
    page_image_path: str,
    bbox_px: list[int],
    out_path: str,
    *,
    margin_px: int = _CROP_MARGIN_PX,
) -> str | None:
    """Crop ``bbox_px`` (plus margin) out of a rendered page image.

    Returns ``None`` on any failure so callers can fall back to the full page.
    """
    try:
        from PIL import Image

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(page_image_path) as image:
            x1, y1, x2, y2 = bbox_px
            box = (
                max(0, int(x1) - margin_px),
                max(0, int(y1) - margin_px),
                min(image.width, int(x2) + margin_px),
                min(image.height, int(y2) + margin_px),
            )
            image.crop(box).save(out_path)
        return out_path
    except Exception as exc:
        logger.warning(
            "[formula] crop failed for {} bbox={}: {}",
            page_image_path,
            bbox_px,
            exc,
        )
        return None


def verify_formulas(
    records: list[FormulaRecord],
    *,
    image_for: Callable[[FormulaRecord], str | None],
    output_dir: str,
    model_name: str | None = None,
    concurrency: int = 4,
    usage_task: str = "parser.formula.verify",
) -> list[FormulaRecord]:
    """Verify every record in a gevent pool; mutates and returns ``records``."""
    if not records:
        return records
    formulas_dir = Path(output_dir) / "formulas"
    formulas_dir.mkdir(parents=True, exist_ok=True)

    import gevent
    from gevent.pool import Pool as GeventPool

    logger.info(
        "[formula] verifying {} formulas (concurrency={})",
        len(records),
        concurrency,
    )
    pool = GeventPool(size=max(1, min(concurrency, len(records))))
    greenlets = [
        pool.spawn(
            _verify_one,
            record=record,
            index=index,
            image_for=image_for,
            formulas_dir=formulas_dir,
            model_name=model_name,
            usage_task=usage_task,
        )
        for index, record in enumerate(records)
    ]
    gevent.joinall(greenlets)
    return records


def _verify_one(
    *,
    record: FormulaRecord,
    index: int,
    image_for: Callable[[FormulaRecord], str | None],
    formulas_dir: Path,
    model_name: str | None,
    usage_task: str,
) -> None:
    rendered_path = formulas_dir / f"formula-{index}.png"
    try:
        render_formula_png(record.latex, str(rendered_path))
    except ValueError as exc:
        record.status = "render_failed"
        logger.info(
            "[formula] render failed, keeping latex ({}): {}",
            exc,
            record.latex,
        )
        return

    source_path = image_for(record)
    if not source_path:
        # No source image to judge against; the LaTeX channel still stands.
        return

    try:
        match, corrected, _reason = _judge_once(
            source_path=source_path,
            rendered_path=str(rendered_path),
            latex=record.latex,
            model_name=model_name,
            usage_task=usage_task,
        )
    except Exception as exc:
        record.status = "judge_failed"
        logger.warning("[formula] judge failed for {}: {}", record.latex, exc)
        return

    if match:
        record.status = "verified"
        return

    corrected = (corrected or "").strip()
    if not corrected or corrected == record.latex:
        record.status = "disputed"
        return

    corrected_path = formulas_dir / f"formula-{index}-corrected.png"
    try:
        render_formula_png(corrected, str(corrected_path))
    except ValueError:
        # The correction does not render; keep the original transcription.
        record.status = "disputed"
        return

    try:
        second_match, _corrected, _reason = _judge_once(
            source_path=source_path,
            rendered_path=str(corrected_path),
            latex=corrected,
            model_name=model_name,
            usage_task=usage_task,
        )
    except Exception as exc:
        record.status = "judge_failed"
        logger.warning("[formula] second judge round failed for {}: {}", corrected, exc)
        return

    if second_match:
        record.original_latex = record.latex
        record.latex = corrected
        record.status = "corrected"
    else:
        # Neither version convinced the judge; restore the original.
        record.status = "disputed"


def _judge_once(
    *,
    source_path: str,
    rendered_path: str,
    latex: str,
    model_name: str | None,
    usage_task: str,
) -> tuple[bool, str, str]:
    """One two-image judge round. Returns ``(match, corrected_latex, reason)``."""
    prompt, temperature, _top_p, max_tokens = build_prompt(
        "formula-verify",
        "",
        "",
        paras={"latex": latex, "max_tokens": 400},
    )

    from shared.services.ai.llm_overrides import get_vision_client

    client, resolved_model = get_vision_client(requested_model=model_name)
    raw_response, _usage = client.chat_completion_with_usage(
        messages=cast(
            Any,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(source_path)},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(rendered_path)},
                        },
                    ],
                }
            ],
        ),
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        usage_task=usage_task,
    )
    data = _loads_lenient(raw_response)
    return (
        bool(data.get("match")),
        str(data.get("corrected_latex") or ""),
        str(data.get("reason") or ""),
    )


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_KNOWN_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _loads_lenient(raw: str) -> dict[str, Any]:
    """``json.loads`` with a repair pass for raw LaTeX backslashes.

    Judge replies embed LaTeX in strict JSON. Models frequently emit raw
    backslashes in two damaging shapes: invalid escapes (``\\sum`` → ``\\s``)
    crash the parse, and worse, ``\\frac``/``\\tag`` parse "successfully" as
    formfeed/tab and silently corrupt the correction. So any response that
    contains a backslash goes through the repair first (re-escaping
    backslashes that introduce LaTeX commands); the plain parse remains the
    fallback for responses without backslashes or when repair cannot fix it.
    """
    if "\\" in raw:
        try:
            return json.loads(_escape_raw_latex_backslashes(raw))
        except json.JSONDecodeError:
            pass
    return json.loads(raw)


def _escape_raw_latex_backslashes(text: str) -> str:
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        nxt = text[index + 1] if index + 1 < length else ""
        if nxt == "\\":  # already-escaped backslash pair: keep verbatim
            out.append("\\\\")
            index += 2
            continue
        if nxt == "u" and index + 5 < length and all(
            c in _HEX_DIGITS for c in text[index + 2 : index + 6]
        ):  # genuine \uXXXX escape
            out.append(text[index : index + 6])
            index += 6
            continue
        if nxt.isalpha():  # raw LaTeX command: escape the backslash
            out.append("\\\\")
            index += 1
            continue
        out.append(char)  # structural escape (\" \n \t …): keep as-is
        index += 1
    return "".join(out)


def _data_url(image_path: str) -> str:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    suffix = Path(image_path).suffix.lstrip(".").lower() or "png"
    return f"data:image/{suffix};base64,{encoded}"
