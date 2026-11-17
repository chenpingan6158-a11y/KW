"""Regex extraction of ``$...$`` / ``$$...$$`` LaTeX formulas from transcribed text."""

from __future__ import annotations

import re
from dataclasses import dataclass

_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
# Bodies that are digits with separators (optionally joined by and/or) are
# currency amounts ("$5", "$1,200", "$5 and $6"), not formulas.
_CURRENCY_BODY_RE = re.compile(r"^\d[\d,.\s]*(?:\b(?:and|or)\b[\d,.\s]*)?$")


@dataclass
class ExtractedFormula:
    """A formula span inside a transcribed text, with byte offsets."""

    latex: str
    display: bool
    start: int
    end: int
    page_num: int | None = None


def extract_latex_formulas(
    text: str, *, page_num: int | None = None
) -> list[ExtractedFormula]:
    """Pull display (``$$..$$``) then inline (``$..$``) formulas out of ``text``.

    Display regions are matched first and masked so the inline pass cannot
    double-match their delimiters. Currency-like bodies are skipped.
    """
    extracted: list[ExtractedFormula] = []
    masked = text
    for match in _DISPLAY_RE.finditer(text):
        body = match.group(1).strip()
        if not _plausible_formula(body):
            continue
        extracted.append(
            ExtractedFormula(
                latex=body,
                display=True,
                start=match.start(),
                end=match.end(),
                page_num=page_num,
            )
        )
        masked = (
            masked[: match.start()]
            + "\x00" * (match.end() - match.start())
            + masked[match.end() :]
        )
    for match in _INLINE_RE.finditer(masked):
        body = match.group(1).strip()
        if not _plausible_formula(body):
            continue
        extracted.append(
            ExtractedFormula(
                latex=body,
                display=False,
                start=match.start(),
                end=match.end(),
                page_num=page_num,
            )
        )
    extracted.sort(key=lambda item: item.start)
    return extracted


def _plausible_formula(body: str) -> bool:
    if not body:
        return False
    return not _CURRENCY_BODY_RE.match(body)


def substitute_formulas(
    text: str, replacements: list[tuple[ExtractedFormula, str]]
) -> str:
    """Rewrite corrected latex back into ``text`` by span, back to front.

    Offsets in ``replacements`` must refer to ``text`` as-is; applying them in
    reverse span order keeps earlier offsets valid. Each formula is re-wrapped
    in its original ``$`` / ``$$`` delimiters.
    """
    result = text
    for formula, corrected in sorted(
        replacements, key=lambda pair: pair[0].start, reverse=True
    ):
        wrapper = "$${}$$" if formula.display else "${}$"
        result = (
            result[: formula.start] + wrapper.format(corrected) + result[formula.end :]
        )
    return result
