# Multimodal Maintenance Intelligence Agent

A grounded, text-first troubleshooting assistant for the ABB ACS580-01 variable-frequency-drive family. It ingests three public manuals, preserves page-level provenance and safety language, combines BM25 and dense retrieval, answers only from retrieved evidence, validates citations, renders highlighted source pages, and abstains when support is weak.

## Architecture

```text
ABB manuals -> PyMuPDF/OCR -> page-bounded chunks -> BM25 in memory
                                      |             -> FastEmbed -> Qdrant
                                      |-> provenance                 |
                                      +-> visual candidate inventory |
Technician question -> BM25 + dense -> reciprocal-rank fusion -> grounded extractive answer
                                                                  |-> validated citations
                                                                  |-> highlighted PDF evidence
                                                                  +-> insufficient-evidence response

Next.js UI -> FastAPI -> PostgreSQL query/audit log
```

The current answer layer is deliberately extractive. It assembles sentences already present in the manuals instead of asking a generative model to invent prose or procedures. Visual candidates from the same validated citation pages are returned in a separate field and displayed as contextual evidence; they cannot affect answer text or abstention decisions.

Operational endpoints expose separate liveness (`/health`) and retrieval-readiness (`/ready`) checks. Every response includes an `X-Request-ID`, API requests are emitted as structured JSON logs, and Prometheus metrics are available at `/metrics`.

The UI and `/evaluation` endpoint expose a versioned evaluation snapshot. Current answer-level results are 100% acceptable-page citation accuracy across 12 answerable questions and 100% abstention recall with a 0% false-answer rate across 8 adversarial unsupported questions. The 20-question, single-product-family benchmark remains intentionally labeled as too small for production-quality conclusions.

## Corpus and generated artifacts

- Source PDFs: `data/manuals/`
- Verified checksums: `evaluation/corpus_checksums.csv`
- Golden questions: `evaluation/golden_questions.csv`
- Documents, pages, chunks, and run manifest: `data/processed/`
- Page-linked raster and vector candidates: `data/processed/visual_assets.jsonl` (generated and ignored by Git)
- Local Qdrant index: `data/qdrant/` (generated and ignored by Git)
- Retrieval reports: `evaluation/results/` (generated and ignored by Git)
- Manually inspected visual labels and metrics: `evaluation/visual_labels.csv` and `evaluation/visual_metrics.json`

Chunks never cross PDF page boundaries. Each retains document name, PDF page, printed page when detected, section path, bounding boxes, extraction method, content type, and safety tags.

## Local setup (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
maintenance-ingest
maintenance-visual-inventory
maintenance-visual-sample
maintenance-visual-evaluate
maintenance-evaluate
maintenance-index --recreate
maintenance-evaluate-full
maintenance-api
```

In a second terminal, install and run the UI:

```powershell
cd web
pnpm install
pnpm dev
```

Open `http://localhost:3000`. The API is at `http://localhost:8000`; interactive API documentation is at `http://localhost:8000/docs`.

## Verification

```powershell
python -m pytest
cd web
pnpm lint
pnpm build
```

## Docker Compose

```powershell
docker compose up --build
```

Compose starts PostgreSQL and Qdrant, runs the one-shot dense indexer, then starts FastAPI and Next.js. Recreating the index intentionally delays the API on first boot. The three manuals and processed chunks must already exist under `data/`.

## Retrieval and grounding behavior

- BM25 handles exact identifiers, parameters, warning codes, and rare terms.
- BGE-small embeddings handle paraphrases and concept similarity.
- Weighted reciprocal-rank fusion merges independently ranked candidate lists without pretending their scores share a scale. The evaluated default favors BM25 16:1 because equal weighting reduced exact-page recall on the current golden set.
- Answer sentences must come from retrieved chunks, and every citation is checked against the chunk registry before returning.
- Visual evidence is joined only after a grounded text answer succeeds, by exact source filename and PDF page. Abstained responses return no visuals, and visual records never enter answer scoring.
- The selected result expands to the immediately adjacent chunks on the same page, improving split procedures and tables without crossing page boundaries; each original chunk remains separately validated and cited.
- Low-confidence retrieval returns an explicit insufficient-evidence response.
- Safety-tagged evidence is surfaced and safety wording is not rewritten into new procedures.

## Visual provenance pipeline

`maintenance-visual-inventory` inventories embedded raster images and vector-heavy page regions before any visual-model experiments. Every record retains the source filename, one-based PDF page, bounding box, dimensions when available, nearby caption text, and a stable asset ID. The current corpus contains 1,462 candidates: 372 raster images and 1,090 vector regions. Optional page crops can be rendered for annotation or later multimodal evaluation.

These are deliberately called *candidates*. The detector does not classify diagram meaning, read wiring relationships, or use visual content to answer questions. That separation makes future diagram experiments measurable and keeps current resume claims honest.

A deterministic 24-item sample covers all six document/type strata (three manuals times raster/vector). Visual inspection found meaningful engineering content in all 24 candidates; the 100% observed sampled precision has a 95% Wilson lower bound of 86.2%. This small, single-family sample measures candidate usefulness only. It does not measure missed visuals (recall), diagram comprehension, or visual question answering, and it has not been independently double-annotated.

The API exposes provenance-checked visual crops at `/visuals/{asset_id}/image`. Arbitrary filenames are never accepted by this route, crop bounds come from the offline inventory, and internal rendered-file paths are excluded from the response schema. The UI labels every crop as contextual evidence that was not used to form the answer.

## Known limitations

- OCR fallback requires a local Tesseract installation; low-text pages are recorded as warnings when it is unavailable.
- Evidence highlighting uses chunk block bounding boxes, not word-perfect character coordinates.
- The extractive answerer is safer and auditable but less conversational than a carefully constrained LLM.
- Qdrant indexing is CPU-heavy on first run because embeddings are created locally.
- The evidence selector is evaluated on only 20 questions from one product family; its perfect answer-level snapshot is a regression baseline, not evidence of generalization.
- Diagram interpretation is not implemented yet; 1,462 page-linked visual candidates have been inventoried, but they are not yet semantically interpreted or used for answering.
- PostgreSQL stores query audit events, while document metadata remains in versioned JSONL artifacts for this MVP.

## Recommended next upgrades

1. Expand and manually verify the golden set, including adversarial and unanswerable questions; tune the abstention threshold from measured precision/coverage.
2. Add reranking after hybrid retrieval and compare it against the existing baseline before accepting the added latency.
3. Add OCR layout regression tests and install Tesseract for scanned manuals.
4. Manually label a representative sample of visual candidates, measure detector precision/recall, then benchmark diagram-aware retrieval against the text-only baseline.
5. Add an optional citation-constrained LLM synthesis layer with sentence-level entailment checks; retain extractive fallback.
6. Pin the Qdrant container image after testing a deployment target, add authentication/TLS, migrations, observability, rate limits, and background indexing.
7. Consider a knowledge graph only if cross-document evaluation demonstrates a measurable failure that graph traversal fixes.
