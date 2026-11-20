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
  (transcribe prompt math rules;                │
   redelimit fallback when delimiters           ▼
   are missing)              render → judge → correct-once → status
                                                │
                    corrected latex spliced back into fine text
                                                │
                    node row extra_metadata["formulas"] → ZIP/DB/MCP
                    (+ extra_metadata["formula_scan"] trace)
```

All three entry paths run the same loop:

- **PDF page path** (C1-C7 scopes) and the **whole_doc shortcut**
  (≤5 pages, no TOC — wired into the same fine stage) via
  `page_memory/page_text_extractor.py`.
- **Image parser** (`formats/image/parser.py`) against the original image
  for standalone formula photos (e.g. a phone picture of one equation).

## Extraction sources

| Source | Formulas | bbox |
|:---|:---|:---|
| `fine_text_extractor=mineru` | `inline_formula` spans → `$..$`; `interline_equation` blocks → own-line `$$..$$` | ✅ layout.json bbox, converted pt→px at the 144-dpi render scale |
| `fine_text_extractor=vlm` | prompt mandates `$..$` / `$$..$$`; regex-extracted from the transcription; **redelimit fallback** (one cheap text rewrite) when the transcription carries math but no `$` delimiters | ❌ (judged against the full page) |
| `fine_text_extractor=ocr` / `none` | none — plain OCR is not trusted for math | — |
| image parser (text branch) | same regex extraction + redelimit fallback | ❌ (the image *is* the source) |

Delimiter compliance of VLM models is unreliable (observed on glm-4v-flash:
~40% of transcriptions flattened the math) — the redelimit fallback
(`redelimit.py`) is what keeps coverage at 100%.

## Verification state machine

Per formula (gevent pool, concurrency-configurable):

1. Render latex via mathtext (normalized: `\begin{array}` wrappers and
   `\tag{}` stripped, `\operatorname*{...}` mapped to built-ins,
   `\\` row separators → newlines). Parse failure → `render_failed`,
   latex kept, VLM skipped.
2. Judge: `(source crop, rendered png)` → strict JSON
   `{match, corrected_latex, reason}` — parsed leniently, because models
   emit raw LaTeX backslashes in both damaging shapes: invalid escapes
   (`\sum` → `\s`) crash the parse, valid-but-wrong ones (`\frac` →
   formfeed) silently corrupt the correction. match → `verified`;
   judge call error → `judge_failed` (latex kept).
3. Mismatch with a correction → re-render corrected → judge once more:
   - match → `corrected` (`original_latex` records what changed)
   - still mismatch, unrenderable correction, or no correction → `disputed`
   with the ORIGINAL latex restored.

On the MinerU path the judge sees a **tight bbox crop** of the formula
from the dpi-144 page render (not the full page) — page sizes come from
the profile's `page_features`; without them every bbox degrades to None
and the judge quietly compares against the whole page.

No failure path drops the original transcription: the floor is always the
pre-change behavior.

`verified` (the boolean) means `status in {verified, corrected}`.

Every chunk that ran the pipeline also carries
`formula_scan: {"found": n, "redelimited"?, "redelimit_failed"?}` so a
zero-formula outcome stays distinguishable from "the pipeline never ran".

## Data model

```json
"formulas": [
  {
    "latex": "S_V(jf) = \\mp j S_u(jf)",
    "status": "corrected",
    "verified": true,
    "page_num": 3,
    "bbox": [326, 978, 1056, 1020],
    "original_latex": "S_V(jf) = + j S_u(jf)"
  }
]
```

`page_num` is present on the PDF path only; `bbox` (4 ints, dpi-144 px)
only on the MinerU path — it anchors the formula on the page render and
enables re-verification without re-parsing; `original_latex` only when
corrected. The fields survive `dataframe_to_chunks` → ZIP `chunks.json`
(validated by `_normalize_formulas` in `shared/services/storage/zip_chunk_schema.py`,
which strips a malformed bbox rather than the formula) →
`document_chunks.chunk_metadata` → MCP `corpus.read` payload.

## Configuration

| Setting | Default | Effect |
|:---|:---|:---|
| `MINERU_ENABLE_FORMULA` | `true` | Ask MinerU for LaTeX formulas. |
| `PAGE_MEMORY_FINE_TEXT_EXTRACTOR` | `none` | Fine-stage extractor (`mineru`/`vlm`/`ocr`/`none`). The PDF formula path needs `mineru` (or `vlm`) here. |
| `FORMULA_VERIFY_ENABLED` | `true` | Master switch for the VLM verification loop (all parsers). When off, formulas stay `unverified` — the LaTeX channel still works. |
| `PAGE_MEMORY_FORMULA_VERIFY_CONCURRENCY` | `4` | Gevent pool size (also per-job via `page_memory_config.formula_verify_concurrency`). |
| `MINERU_RESULT_PROXY` | *(empty)* | Forward proxy for MinerU result-zip downloads. The result CDN resets OpenSSL ClientHellos and container (vpnkit) TCP paths; downloads use a browser TLS fingerprint and, when set, egress through this host-side proxy. |

Cost note: verification costs 1–2 vision calls per formula (user-accepted:
all formulas, no density gating). `FORMULA_VERIFY_ENABLED=false` is the
kill-switch; `fine_text_extractor=none` (the default) keeps the whole PDF
formula path dormant, exactly as before this feature.

## Modules

- `models.py` — `FormulaRecord`, status set, `to_metadata()` (incl. bbox).
- `extractor.py` — `extract_latex_formulas` (display-first, currency filter) /
  `substitute_formulas` (back-to-front span rewrite).
- `redelimit.py` — math-signal detector + one-shot LLM rewrite restoring
  `$..$` delimiters on non-compliant transcriptions.
- `renderer.py` — mathtext rendering (Agg, lazy import): `\text{}`→`\mathrm{}`,
  strips `\displaystyle`/`\label`/`\tag` and `\begin{array}` wrappers,
  maps `\operatorname*{...}` onto built-ins, `\\` row separators → newlines.
- `verifier.py` — `verify_formulas` pool, `crop_bbox`, two-image `_judge_once`
  with a lenient JSON parse for raw LaTeX backslashes.

## Testing

- Unit: `apps/worker/tests/unit/test_formula_extractor.py`,
  `test_formula_renderer.py`, `test_formula_verifier.py` (incl. lenient
  JSON), `test_formula_redelimit.py`, `test_formula_models.py` (bbox),
  `test_skeleton_trailing_pages.py`.
- Contract: cases in `test_page_text_extractor_contract.py` (layout walker
  formula collection, bbox ×2 scaling, vlm `$..$` extraction),
  `test_image_parse_contract.py` (formula metadata + redelimit fallback +
  scan on image chunks), `test_whole_doc_formula_contract.py` (whole_doc
  fine-stage wiring).
- Shared: `test_page_memory_config.py`, `test_chunk_relations.py` (ZIP
  whitelist keeps valid formulas/bbox/scan, drops malformed).

## Out of scope (follow-ups)

- DOCX OMML equations (`formats/docx/parser.py` TODO) still dropped.
- Formulas are metadata riders, not first-class chunks — no `connect_to` edges,
  no per-formula addressing, no cross-document LaTeX-hash dedup yet.
- No formula-density gating for verification cost.
- MinerU inline-formula recall is limited (a deliberately inline-dense test
  page yielded zero spans); accepted for now rather than adding a
  mixed-source VLM re-extraction pass.
