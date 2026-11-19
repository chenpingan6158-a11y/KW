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
    # Crop box on the dpi-144 page render (MinerU layout.json, pt→px); rides
    # metadata as "bbox" so the record stays re-verifiable and UI-anchorable.
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
        if self.bbox_px:
            # MinerU-only provenance (layout.json pt→px on the dpi-144 render):
            # anchors the formula on the page image and enables re-verification
            # without re-parsing. VLM / image-parser records never set it.
            payload["bbox"] = [
                int(round(value)) for value in self.bbox_px[:4]
            ]
        return payload
