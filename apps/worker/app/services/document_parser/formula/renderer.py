"""Render LaTeX formulas to PNG via matplotlib mathtext (Agg, no LaTeX install)."""

from __future__ import annotations

import re

_TEXT_CMD_RE = re.compile(r"\\text\s*\{")
# Commands mathtext cannot parse and that carry no visual meaning here.
_STRIP_CMDS_RE = re.compile(r"\\(?:displaystyle|label|tag)\s*(?:\{[^{}]*\})?")
# mathtext separates rows by newlines inside cases/array environments, while
# LLM output uses the standard LaTeX ``\\`` row separator.
_ROW_SEP_RE = re.compile(r"\\\\\s*")
# MinerU emits \operatorname* { l i m } (spaced letters, starred); mathtext
# has neither \operatorname nor the star. Map known operator names to their
# built-in commands, everything else to \mathrm.
_OPNAME_RE = re.compile(r"\\operatorname\*?\s*\{\s*([^{}]*?)\s*\}")
_KNOWN_OPS = frozenset(
    {
        "lim", "max", "min", "inf", "sup", "sin", "cos", "tan", "cot",
        "sec", "csc", "ln", "log", "exp", "det", "dim", "deg", "gcd",
        "arg", "arctan", "arcsin", "arccos", "Re", "Im",
    }
)
# mathtext has no array environment; MinerU wraps multi-line formulas in
# \begin{array}{r}...\end{array}. Strip the wrapper, keep the rows (the
# row-separator pass turns \\ into newlines, matching how cases renders).
_ARRAY_BEGIN_RE = re.compile(r"\\begin\s*\{\s*array\s*\}\s*\{[^{}]*\}")
_ARRAY_END_RE = re.compile(r"\\end\s*\{\s*array\s*\}")


def render_formula_png(latex: str, output_path: str, *, dpi: int = 200) -> str:
    """Render ``latex`` (without ``$`` delimiters) to a tightly-cropped PNG.

    Raises ``ValueError`` when mathtext cannot parse the sanitized expression;
    callers map that to a ``render_failed`` formula status.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    expression = _sanitize(latex)
    fig = plt.figure(figsize=(8, 2), facecolor="white")
    try:
        fig.text(
            0.5,
            0.5,
            f"${expression}$",
            fontsize=22,
            color="black",
            ha="center",
            va="center",
        )
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="white",
        )
    except Exception as exc:
        raise ValueError(f"mathtext failed to render formula: {exc}") from exc
    finally:
        plt.close(fig)
    return output_path


def _sanitize(latex: str) -> str:
    expression = latex.strip().strip("$").strip()
    expression = _TEXT_CMD_RE.sub(r"\\mathrm{", expression)
    expression = _STRIP_CMDS_RE.sub("", expression)
    expression = _OPNAME_RE.sub(_normalize_opname, expression)
    expression = _ARRAY_BEGIN_RE.sub("", expression)
    expression = _ARRAY_END_RE.sub("", expression)
    expression = _ROW_SEP_RE.sub("\n", expression)
    return expression


def _normalize_opname(match: re.Match[str]) -> str:
    name = "".join(match.group(1).split())
    if name in _KNOWN_OPS:
        return "\\" + name
    return "\\mathrm{" + name + "}"
