"""Unit tests for the zero-extraction redelimit fallback."""

from __future__ import annotations

from typing import Any

from app.services.document_parser.formula.redelimit import (
    looks_mathematical,
    redelimit_transcription,
)

# Real shapes observed from glm-4.1v-thinking-flash / glm-4v-flash runs.
_FLAT_MATH = (
    "U(t) = A(t)cosΦ(t) = A(t)cos[ω₀t-φ(t)]\n"
    "A(t) = [U(t)^2 + V(t)^2]^(1/2)\n"
    "Φ(t) = ω₀t - φ(t) = arctan[V(t)/U(t)]"
)
_BARE_LATEX = "Re{U(t)}=\\frac{1}{2}[U(t)+U*(t)] 且 ω(t)=ω₀"
_PLAIN_PROSE = "今天天气很好,我们去公园散步,顺便聊了聊最近的工作和生活。"
_ENGLISH_PROSE = (
    "The quarterly report shows revenue grew, but the summary paragraph "
    "mentions no numbers worth extracting here."
)


def test_looks_mathematical_true_for_flattened_math() -> None:
    # operators (=, ^) + greek (ω, φ, Φ) + functions (cos, arctan)
    assert looks_mathematical(_FLAT_MATH)


def test_looks_mathematical_true_for_bare_latex() -> None:
    # latex commands (\frac) + greek (ω)
    assert looks_mathematical(_BARE_LATEX)


def test_looks_mathematical_false_for_prose() -> None:
    assert not looks_mathematical(_PLAIN_PROSE)
    assert not looks_mathematical(_ENGLISH_PROSE)


def test_looks_mathematical_false_for_empty() -> None:
    assert not looks_mathematical("")
    assert not looks_mathematical("   ")


def test_looks_mathematical_requires_two_signal_categories() -> None:
    # One isolated "=" alone is not math.
    assert not looks_mathematical("revenue = cost")


def _patch_llm(
    monkeypatch: Any,
    completion: Any,
    *,
    prompt_sink: dict[str, Any] | None = None,
) -> None:
    """Patch the lazy-imported helpers at their source modules."""

    class _FakeClient:
        def chat_completion(self, **kwargs: Any) -> Any:
            del kwargs
            if isinstance(completion, Exception):
                raise completion
            return completion

    import shared.services.ai.llm_overrides as llm_overrides
    import shared.services.ai.prompt_service as prompt_service

    monkeypatch.setattr(
        llm_overrides,
        "get_text_client",
        lambda: (_FakeClient(), "fake-text-model"),
    )

    def _fake_build_prompt(task: str, *args: Any, **kwargs: Any) -> Any:
        if prompt_sink is not None:
            prompt_sink["task"] = task
        return ("prompt", 0.1, 0.1, 2000)

    monkeypatch.setattr(prompt_service, "build_prompt", _fake_build_prompt)


def test_redelimit_returns_stripped_rewrite(monkeypatch: Any) -> None:
    sink: dict[str, Any] = {}
    _patch_llm(monkeypatch, "  $A(t) = \\omega_0 t$  ", prompt_sink=sink)

    result = redelimit_transcription("A(t) = ω₀t")

    assert result == "$A(t) = \\omega_0 t$"
    assert sink["task"] == "formula-redelimit"


def test_redelimit_returns_none_on_client_error(monkeypatch: Any) -> None:
    _patch_llm(monkeypatch, RuntimeError("provider down"))
    assert redelimit_transcription("A(t) = ω₀t") is None


def test_redelimit_returns_none_for_empty_input(monkeypatch: Any) -> None:
    assert redelimit_transcription("") is None


def test_redelimit_returns_none_when_unchanged(monkeypatch: Any) -> None:
    _patch_llm(monkeypatch, "A(t) = ω₀t")  # identical to input
    assert redelimit_transcription("A(t) = ω₀t") is None
