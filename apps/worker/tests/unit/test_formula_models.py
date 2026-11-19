"""Unit tests for FormulaRecord metadata serialization (incl. bbox)."""

from __future__ import annotations

from app.services.document_parser.formula.models import FormulaRecord


def test_to_metadata_omits_unset_optionals() -> None:
    record = FormulaRecord(latex="E=mc^2")
    assert record.to_metadata() == {
        "latex": "E=mc^2",
        "status": "unverified",
        "verified": False,
    }


def test_to_metadata_serializes_mineru_bbox() -> None:
    record = FormulaRecord(
        latex="E=mc^2",
        status="verified",
        page_num=3,
        bbox_px=[120.4, 88.6, 540.0, 152.9],
    )
    payload = record.to_metadata()
    assert payload["bbox"] == [120, 89, 540, 153]
    assert payload["page_num"] == 3
    assert payload["verified"] is True


def test_to_metadata_keeps_bbox_with_corrected_original() -> None:
    record = FormulaRecord(
        latex="S_v(jf) = -j S_u(jf)",
        status="corrected",
        original_latex="S_v(jf) = +j S_u(jf)",
        bbox_px=[0, 10, 100, 60],
    )
    payload = record.to_metadata()
    assert payload["bbox"] == [0, 10, 100, 60]
    assert payload["original_latex"] == "S_v(jf) = +j S_u(jf)"
