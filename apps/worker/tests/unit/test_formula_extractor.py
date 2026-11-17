from __future__ import annotations

from app.services.document_parser.formula.extractor import (
    extract_latex_formulas,
    substitute_formulas,
)


def test_display_not_double_matched_by_inline_pass() -> None:
    text = "intro $$E = mc^2$$ tail and inline $a+b$ end"
    formulas = extract_latex_formulas(text)

    assert [item.latex for item in formulas] == ["E = mc^2", "a+b"]
    assert [item.display for item in formulas] == [True, False]


def test_extract_skips_currency_dollars() -> None:
    text = "cost $5 and $6 total, price $1,200 only"
    assert extract_latex_formulas(text) == []


def test_extract_keeps_digit_leading_formulas() -> None:
    formulas = extract_latex_formulas("so $2x + 1 = 0$ holds")
    assert [item.latex for item in formulas] == ["2x + 1 = 0"]


def test_extract_carries_page_num() -> None:
    formulas = extract_latex_formulas("value $q$", page_num=7)
    assert formulas[0].page_num == 7


def test_substitute_back_to_front_offsets() -> None:
    text = "a $x+y$ b $$z_1$$ c $w$ d"
    formulas = extract_latex_formulas(text)
    replacements = [(formulas[0], "x-y"), (formulas[2], "w^2")]

    result = substitute_formulas(text, replacements)

    assert "$x-y$" in result
    assert "$w^2$" in result
    assert "$$z_1$$" in result
    assert "$x+y$" not in result
    assert "$w$" not in result
