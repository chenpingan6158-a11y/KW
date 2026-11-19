from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from app.services.document_parser.formula.renderer import render_formula_png


def test_render_quadratic_formula_produces_png(tmp_path: Path) -> None:
    out = tmp_path / "quadratic.png"
    render_formula_png(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", str(out))
    assert out.stat().st_size > 0


def test_render_strips_dollar_delimiters(tmp_path: Path) -> None:
    out = tmp_path / "dollars.png"
    render_formula_png("$$E = mc^2$$", str(out))
    assert out.stat().st_size > 0


def test_render_unsupported_latex_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_formula_png(
            r"\begin{align} x &= 1 \end{align}",
            str(tmp_path / "align.png"),
        )


def test_row_separator_backslashes_render_like_newlines(tmp_path):
    """LLM ``\\`` row separators must render the same as newline rows."""
    from app.services.document_parser.formula.renderer import render_formula_png

    latex = (
        "S_v(jf)=\\begin{cases}-jS_u(jf), & f>0\\\\ jS_u(jf), & f<0\\end{cases}"
    )
    out = render_formula_png(latex, str(tmp_path / "cases.png"))
    assert (tmp_path / "cases.png").exists() and out


def test_mineru_array_wrapper_renders(tmp_path):
    """MinerU wraps multi-line formulas in \\begin{array}{r}; strip and render."""
    latex = (
        "\\begin{array} { r } { \\omega _ { 0 } = 2 \\pi f _ { 0 } , "
        "\\qquad \\varphi _ { V } ( f ) = \\varphi _ { u } ( f ) "
        "+ \\frac { \\pi } { 2 } } \\end{array}\\tag{2.c}"
    )
    out = render_formula_png(latex, str(tmp_path / "array.png"))
    assert (tmp_path / "array.png").stat().st_size > 0 and out


def test_starred_spaced_operatorname_maps_to_builtin(tmp_path):
    """\\operatorname* { l i m } (spaced, starred) becomes \\lim."""
    latex = (
        "\\operatorname* { l i m } _ { n \\to \\infty } "
        "( 1 + { \\frac { 1 } { n } } ) ^ { n } = e\\tag{5.a}"
    )
    out = render_formula_png(latex, str(tmp_path / "lim.png"))
    assert (tmp_path / "lim.png").stat().st_size > 0 and out


def test_unknown_operatorname_falls_back_to_mathrm(tmp_path):
    latex = r"\operatorname{tr}(A) = 3"
    out = render_formula_png(latex, str(tmp_path / "tr.png"))
    assert (tmp_path / "tr.png").stat().st_size > 0 and out
