from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("TMP_PATH", "/tmp/knowhere-test")
os.environ.setdefault("S3_BUCKET_NAME", "test-uploads")
os.environ.setdefault("S3_ACCESS_KEY_ID", "test")
os.environ.setdefault("S3_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("S3_TEMP_PATH", "/tmp")

import app.services.document_parser.formula.verifier as verifier_module
from app.services.document_parser.formula.models import FormulaRecord


def _always_render(latex: str, path: str, **_kwargs: object) -> str:
    del latex
    return path


def test_verify_match_first_round(monkeypatch, tmp_path) -> None:
    judge_calls: list[str] = []

    def fake_judge(**kwargs):
        judge_calls.append(kwargs["latex"])
        return True, "", "identical"

    monkeypatch.setattr(verifier_module, "_judge_once", fake_judge)
    monkeypatch.setattr(verifier_module, "render_formula_png", _always_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "verified"
    assert result[0].verified is True
    assert judge_calls == ["x = y"]


def test_verify_correction_round_two_match(monkeypatch, tmp_path) -> None:
    judge_calls: list[str] = []

    def fake_judge(**kwargs):
        judge_calls.append(kwargs["latex"])
        if len(judge_calls) == 1:
            return False, "x \\mp y", "sign differs"
        return True, "", "now identical"

    monkeypatch.setattr(verifier_module, "_judge_once", fake_judge)
    monkeypatch.setattr(verifier_module, "render_formula_png", _always_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "corrected"
    assert result[0].latex == "x \\mp y"
    assert result[0].original_latex == "x = y"
    assert result[0].verified is True
    assert judge_calls == ["x = y", "x \\mp y"]


def test_verify_disputed_after_two_mismatches(monkeypatch, tmp_path) -> None:
    def fake_judge(**kwargs):
        del kwargs
        return False, "x \\mp y", "still differs"

    monkeypatch.setattr(verifier_module, "_judge_once", fake_judge)
    monkeypatch.setattr(verifier_module, "render_formula_png", _always_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "disputed"
    assert result[0].latex == "x = y"
    assert result[0].original_latex is None
    assert result[0].verified is False


def test_verify_mismatch_without_correction_is_disputed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        verifier_module,
        "_judge_once",
        lambda **kwargs: (False, "", "unreadable source"),
    )
    monkeypatch.setattr(verifier_module, "render_formula_png", _always_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "disputed"
    assert result[0].latex == "x = y"


def test_corrected_render_failure_keeps_original(monkeypatch, tmp_path) -> None:
    def fake_render(latex: str, path: str, **_kwargs: object) -> str:
        if latex == "x \\mp y":
            raise ValueError("mathtext failed")
        return path

    monkeypatch.setattr(
        verifier_module,
        "_judge_once",
        lambda **kwargs: (False, "x \\mp y", "sign differs"),
    )
    monkeypatch.setattr(verifier_module, "render_formula_png", fake_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "disputed"
    assert result[0].latex == "x = y"
    assert result[0].original_latex is None


def test_render_failure_skips_vlm(monkeypatch, tmp_path) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("judge must not run when rendering fails")

    monkeypatch.setattr(verifier_module, "_judge_once", boom)

    def failing_render(latex: str, path: str, **_kwargs: object) -> str:
        raise ValueError("mathtext failed")

    monkeypatch.setattr(verifier_module, "render_formula_png", failing_render)

    records = [FormulaRecord(latex="x = \\unknowncmd y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "render_failed"
    assert result[0].latex == "x = \\unknowncmd y"
    assert result[0].verified is False


def test_judge_exception_marks_judge_failed(monkeypatch, tmp_path) -> None:
    def failing_judge(**_kwargs):
        raise RuntimeError("vision client down")

    monkeypatch.setattr(verifier_module, "_judge_once", failing_judge)
    monkeypatch.setattr(verifier_module, "render_formula_png", _always_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: "/tmp/source.png",
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "judge_failed"
    assert result[0].latex == "x = y"


def test_missing_source_image_stays_unverified(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(verifier_module, "render_formula_png", _always_render)

    records = [FormulaRecord(latex="x = y")]
    result = verifier_module.verify_formulas(
        records,
        image_for=lambda _record: None,
        output_dir=str(tmp_path),
        concurrency=1,
    )

    assert result[0].status == "unverified"


def test_loads_lenient_repairs_raw_latex_backslashes() -> None:
    from app.services.document_parser.formula.verifier import _loads_lenient

    # Invalid escape (backslash-s in \sum) would crash plain json.loads.
    raw = '{"match": false, "corrected_latex": "\\sum_{k=1}^n k^2", "reason": "ok"}'
    data = _loads_lenient(raw)
    assert data["corrected_latex"] == "\\sum_{k=1}^n k^2"

    # Valid-but-corrupting escape: \frac's \f would parse as formfeed.
    raw = '{"match": true, "corrected_latex": null, "reason": "needs \\frac a b"}'
    data = _loads_lenient(raw)
    assert data["reason"] == "needs \\frac a b"

    # \tag's \t would parse as a tab and silently corrupt the latex.
    raw = '{"match": false, "corrected_latex": "x=1\\tag{2.b}", "reason": "r"}'
    data = _loads_lenient(raw)
    assert data["corrected_latex"] == "x=1\\tag{2.b}"


def test_loads_lenient_keeps_valid_json_verbatim() -> None:
    from app.services.document_parser.formula.verifier import _loads_lenient

    # Already-escaped LaTeX (JSON double-backslash frac) stays single.
    raw = '{"match": false, "corrected_latex": "\\\\frac{1}{2}", "reason": "x"}'
    assert _loads_lenient(raw)["corrected_latex"] == "\\frac{1}{2}"

    # Structural escapes and genuine \uXXXX pass through untouched.
    raw = '{"match": true, "corrected_latex": null, "reason": "line1\\nline2 \\u4f60"}'
    # repair keeps JSON newline escapes literal; unicode escapes still decode
    assert _loads_lenient(raw)["reason"] == "line1\\nline2 \u4f60"
