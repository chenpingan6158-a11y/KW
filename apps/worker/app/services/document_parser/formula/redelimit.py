"""Zero-extraction fallback: redelimit delimiter-less math via one LLM rewrite.

The extractor only matches ``$..$`` / ``$$..$$`` spans, but VLM transcriptions
do not reliably follow the delimiter instruction (observed: glm-4v-flash
complied for one image and flattened the next three). When extraction comes
back empty but the transcription carries clear math signals, one cheap
text-to-text call rewrites the same transcription with delimiters so the
render-back loop gets a second chance. The rewrite never invents content: if
it yields no formulas either, the original transcription stands.
"""

from __future__ import annotations

import re

from loguru import logger

# Signal categories whose co-occurrence marks a text as math-bearing. Prose
# words that double as math names (max/min/log/exp) are deliberately excluded.
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_OPERATOR_RE = re.compile(r"[=^_±×÷≤≥≠→∑∫]")
_FUNCTION_RE = re.compile(
    r"\b(?:cos|sin|tan|cot|sec|csc|arctan|arcsin|arccos|ln|det|arg)\b",
    re.IGNORECASE,
)
_GREEK_RE = re.compile(r"[ωφψπθαβγδλμσΩΔΦΠ]")
_SCRIPT_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ₀₁₂₃₄₅₆₇₈₉]")
_MATH_SIGNAL_RES = (
    _LATEX_COMMAND_RE,
    _OPERATOR_RE,
    _FUNCTION_RE,
    _GREEK_RE,
    _SCRIPT_RE,
)

_MIN_SIGNAL_CATEGORIES = 2


def looks_mathematical(text: str) -> bool:
    """Whether ``text`` carries enough math signals to justify a rewrite."""
    if not text:
        return False
    hits = sum(1 for pattern in _MATH_SIGNAL_RES if pattern.search(text))
    return hits >= _MIN_SIGNAL_CATEGORIES


def redelimit_transcription(
    text: str,
    *,
    usage_task: str = "parser.formula.redelimit",
) -> str | None:
    """Rewrite ``text`` so formulas carry ``$`` / ``$$`` delimiters.

    Returns the rewritten transcription, or ``None`` when the call fails or
    the model returns nothing usable. Callers re-run the extractor on the
    result; a rewrite that still yields no formulas is discarded.
    """
    if not text:
        return None
    from shared.services.ai.llm_overrides import get_text_client
    from shared.services.ai.prompt_service import build_prompt

    prompt, temperature, top_p, max_tokens = build_prompt(
        "formula-redelimit",
        text,
        "",
    )
    try:
        client, model = get_text_client()
        rewritten = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            usage_task=usage_task,
        )
    except Exception as exc:
        logger.warning("[formula] redelimit rewrite failed: {}", exc)
        return None
    if not isinstance(rewritten, str):
        return None
    rewritten = rewritten.strip()
    if not rewritten or rewritten == text:
        return None
    return rewritten
