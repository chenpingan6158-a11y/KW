"""Formula data model shared by the LaTeX extraction and verification stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FORMULA_STATUSES = frozenset(
    {
        "unverified",
        "verified",
        "corrected",
        "disputed",
        "render_failed",
        "judge_failed",
    }
)


@dataclass
class FormulaRecord:
    """One extracted formula travelling through the parse pipeline."""

    latex: str
    status: str = "unverified"
    page_num: int | None = None
    original_latex: str | None = None
    # Pipeline-internal crop box on the dpi-144 page render; never serialized.
    bbox_px: list[int] | None = None

    @property
    def verified(self) -> bool:
        return self.status in {"verified", "corrected"}

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "latex": self.latex,
            "status": self.status,
            "verified": self.verified,
        }
        if self.page_num is not None:
            payload["page_num"] = self.page_num
        if self.original_latex is not None:
            payload["original_latex"] = self.original_latex
        return payload
