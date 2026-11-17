"""Render LaTeX formulas to PNG via matplotlib mathtext (Agg, no LaTeX install)."""

from __future__ import annotations

import re

_TEXT_CMD_RE = re.compile(r"\\text\s*\{")
# Commands mathtext cannot parse and that carry no visual meaning here.
_STRIP_CMDS_RE = re.compile(r"\\(?:displaystyle|label|tag)\s*(?:\{[^{}]*\})?")


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
    return expression
