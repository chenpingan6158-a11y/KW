"""Contract tests for the whole_doc fine-stage formula wiring."""

from __future__ import annotations

from typing import Any

from shared.services.chunks.dataframe_chunk_converter import dataframe_to_chunks


def _formula_records() -> list[Any]:
    from app.services.document_parser.formula.models import FormulaRecord

    return [
        FormulaRecord(latex="E=mc^2", status="verified", page_num=1),
        FormulaRecord(
            latex="S_v(jf) = -j S_u(jf)",
            status="corrected",
            page_num=2,
            original_latex="S_v(jf) = +j S_u(jf)",
        ),
    ]


def test_whole_doc_dataframe_splices_fine_text_and_formula_metadata(
    monkeypatch: Any,
) -> None:
    import app.services.page_memory.memory_service as memory_service

    records = _formula_records()
    fine_text = {
        1: "page one with $E=mc^2$ inline",
        2: "page two $$S_v(jf) = -j S_u(jf)$$ display",
    }

    def fake_fine_stage(**kwargs: Any) -> tuple[dict[int, str], dict[int, list[Any]], dict[str, Any]]:
        assert kwargs["pdf_path"] == "/tmp/doc.pdf"
        return fine_text, {1: [records[0]], 2: [records[1]]}, {"found": 2}

    monkeypatch.setattr(memory_service, "_run_whole_doc_fine_stage", fake_fine_stage)

    dataframe = memory_service._build_whole_doc_dataframe(
        filename="whole.pdf",
        page_count=2,
        verdict="whole_doc",
        page_texts={1: "RAW page one", 2: "RAW page two"},
        body_text_by_page={},
        pdf_path="/tmp/doc.pdf",
        output_dir="/tmp/out",
        page_memory_config=memory_service.PageMemoryConfig(),
        vlm_model=None,
    )
    chunk = dataframe_to_chunks(dataframe)[0]

    # Fine text replaces the raw extractor text page by page.
    assert "$E=mc^2$ inline" in chunk["content"]
    assert "$$S_v(jf) = -j S_u(jf)$$ display" in chunk["content"]
    assert "RAW page one" not in chunk["content"]

    formulas = chunk["metadata"]["formulas"]
    assert [item["latex"] for item in formulas] == ["E=mc^2", "S_v(jf) = -j S_u(jf)"]
    assert formulas[1]["original_latex"] == "S_v(jf) = +j S_u(jf)"
    assert formulas[0]["page_num"] == 1
    assert chunk["metadata"]["formula_scan"] == {"found": 2}


def test_whole_doc_dataframe_without_fine_stage_keeps_raw_text() -> None:
    """Legacy callers (no pdf/output/config) keep the pre-wiring behavior."""
    import app.services.page_memory.memory_service as memory_service

    dataframe = memory_service._build_whole_doc_dataframe(
        filename="whole.pdf",
        page_count=2,
        verdict="whole_doc",
        page_texts={1: "RAW page one", 2: "RAW page two"},
        body_text_by_page={},
    )
    chunk = dataframe_to_chunks(dataframe)[0]

    assert "RAW page one" in chunk["content"]
    assert "formulas" not in chunk["metadata"]
    assert "formula_scan" not in chunk["metadata"]


def test_run_whole_doc_fine_stage_skips_when_extractor_none() -> None:
    import app.services.page_memory.memory_service as memory_service

    fine_text, formulas, scan = memory_service._run_whole_doc_fine_stage(
        pdf_path="/tmp/doc.pdf",
        page_count=2,
        pages=[1, 2],
        output_dir="/tmp/out",
        page_memory_config=memory_service.PageMemoryConfig(
            fine_text_extractor="none"
        ),
        vlm_model=None,
    )
    assert fine_text == {}
    assert formulas == {}
    assert scan == {"found": 0}
