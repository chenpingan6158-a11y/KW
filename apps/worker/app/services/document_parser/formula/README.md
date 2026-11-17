# Formula LaTeX extraction & render-back verification

This package gives math formulas a structured, verifiable representation in the
parse pipeline. Before it existed, formulas were flattened into natural-language
VLM transcriptions with no validation — a transcribed `\mp` that came back as
`+` entered the corpus silently wrong, and downstream agents had no way to know.

## What it does

1. **LaTeX channel** — formulas enter the corpus as LaTeX: inline `$...$` stays
   in chunk content (searchable by BM25/grep), and a structured list rides
   `metadata.formulas`.
2. **Render-back verification** — every extracted formula is rendered to a PNG
   (matplotlib mathtext), judged against the original page crop / image by a
   two-image VLM prompt (`formula-verify`), and corrected at most once when the
   judge reads something different off the source.

## Pipeline position

```
fine text extraction                      verification loop (new)
─────────────────────                     ────────────────────────
mineru extractor  → layout.json formulas  → FormulaRecord
  (enable_formula, bbox pt→px)                  │
vlm extractor     → $..$ in transcription→ FormulaRecord (no bbox)
  (transcribe prompt math rules)                │
                                                ▼
                     render → judge → correct-once → status
                                                │
                    corrected latex spliced back into fine text
                                                │
                    node row extra_metadata["formulas"] → ZIP/DB/MCP
```

The image parser (`formats/image/parser.py`) runs the same extraction +
verification against the original image for standalone formula photos
(e.g. a phone picture of one equation).

## Extraction sources

| Source | Formulas | bbox |
|:---|:---|:---|
| `fine_text_extractor=mineru` | `inline_formula` spans → `$..$`; `interline_equation` blocks → own-line `$$..$$` | ✅ layout.json bbox, converted pt→px at the 144-dpi render scale |
| `fine_text_extractor=vlm` | prompt mandates `$..$` / `$$..$$`; regex-extracted from the transcription | ❌ (judged against the full page) |
| `fine_text_extractor=ocr` / `none` | none — plain OCR is not trusted for math | — |
| image parser (text branch) | same regex extraction over the transcription | ❌ (the image *is* the source) |

## Verification state machine

Per formula (gevent pool, concurrency-configurable):

1. Render latex via mathtext. Parse failure → `render_failed`, latex kept, VLM skipped.
2. Judge: `(source crop, rendered png)` → strict JSON `{match, corrected_latex, reason}`.
   - match → `verified`
   - judge call error → `judge_failed` (latex kept)
3. Mismatch with a correction → re-render corrected → judge once more:
   - match → `corrected` (`original_latex` records what changed)
   - still mismatch, unrenderable correction, or no correction → `disputed`
   with the ORIGINAL latex restored.

No failure path drops the original transcription: the floor is always the
pre-change behavior.

`verified` (the boolean) means `status in {verified, corrected}`.

## Data model

```json
"formulas": [
  {
    "latex": "S_V(jf) = \\mp j S_u(jf)",
    "status": "corrected",
    "verified": true,
    "page_num": 3,
    "original_latex": "S_V(jf) = + j S_u(jf)"
  }
]
```

`page_num` is present on the PDF path only; `original_latex` only when
corrected. The field survives `dataframe_to_chunks` → ZIP `chunks.json`
(validated by `_normalize_formulas` in `shared/services/storage/zip_chunk_schema.py`)
→ `document_chunks.chunk_metadata` → MCP `corpus.read` payload, unchanged.

## Configuration

| Setting | Default | Effect |
|:---|:---|:---|
| `MINERU_ENABLE_FORMULA` | `true` | Ask MinerU for LaTeX formulas. |
| `FORMULA_VERIFY_ENABLED` | `true` | Master switch for the VLM verification loop (all parsers). When off, formulas stay `unverified` — the LaTeX channel still works. |
| `PAGE_MEMORY_FORMULA_VERIFY_CONCURRENCY` | `4` | Gevent pool size (also per-job via `page_memory_config.formula_verify_concurrency`). |

Cost note: verification costs 1–2 vision calls per formula (user-accepted:
all formulas, no density gating). `FORMULA_VERIFY_ENABLED=false` is the
kill-switch; `fine_text_extractor=none` (the default) keeps the whole PDF
formula path dormant, exactly as before this feature.

## Modules

- `models.py` — `FormulaRecord`, status set, `to_metadata()`.
- `extractor.py` — `extract_latex_formulas` (display-first, currency filter) /
  `substitute_formulas` (back-to-front span rewrite).
- `renderer.py` — mathtext rendering (Agg, lazy import, `\text{}`→`\mathrm{}`,
  strips `\displaystyle`/`\label`/`\tag`).
- `verifier.py` — `verify_formulas` pool, `crop_bbox`, two-image `_judge_once`.

## Testing

- Unit: `apps/worker/tests/unit/test_formula_extractor.py`,
  `test_formula_renderer.py`, `test_formula_verifier.py`.
- Contract: new cases in `test_page_text_extractor_contract.py` (layout walker
  formula collection, bbox ×2 scaling, vlm `$..$` extraction) and
  `test_image_parse_contract.py` (formula metadata on image chunks).
- Shared: `test_page_memory_config.py`, `test_chunk_relations.py` (ZIP
  whitelist keeps valid formulas, drops malformed).

## Out of scope (follow-ups)

- DOCX OMML equations (`formats/docx/parser.py` TODO) still dropped.
- Formulas are metadata riders, not first-class chunks — no `connect_to` edges,
  no per-formula addressing, no cross-document LaTeX-hash dedup yet.
- No formula-density gating for verification cost.
