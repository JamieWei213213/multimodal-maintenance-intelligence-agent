# Multimodal Maintenance Intelligence Agent

A grounded, text-first troubleshooting assistant for the ABB ACS580-01 variable-frequency-drive family. It ingests three public manuals, preserves page-level provenance and safety language, combines BM25 and dense retrieval, answers only from retrieved evidence, validates citations, renders highlighted source pages, and abstains when support is weak.

## Architecture

```text
ABB manuals -> PyMuPDF/OCR -> page-bounded chunks -> BM25 in memory
                                      |             -> FastEmbed -> Qdrant
                                      +-> provenance                 |
Technician question -> BM25 + dense -> reciprocal-rank fusion -> grounded extractive answer
                                                                  |-> validated citations
                                                                  |-> highlighted PDF evidence
                                                                  +-> insufficient-evidence response

Next.js UI -> FastAPI -> PostgreSQL query/audit log
```

The current answer layer is deliberately extractive. It assembles sentences already present in the manuals instead of asking a generative model to invent prose or procedures.

Operational endpoints expose separate liveness (`/health`) and retrieval-readiness (`/ready`) checks. Every response includes an `X-Request-ID`, API requests are emitted as structured JSON logs, and Prometheus metrics are available at `/metrics`.

The UI and `/evaluation` endpoint expose a versioned evaluation snapshot. Current answer-level results are 100% expected-page citation accuracy and 100% abstention recall on the initial five-question set. These numbers are intentionally displayed with a small-sample warning and must not be treated as production-quality estimates.

## Corpus and generated artifacts

- Source PDFs: `data/manuals/`
- Verified checksums: `evaluation/corpus_checksums.csv`
- Golden questions: `evaluation/golden_questions.csv`
- Documents, pages, chunks, and run manifest: `data/processed/`
- Local Qdrant index: `data/qdrant/` (generated and ignored by Git)
- Retrieval reports: `evaluation/results/` (generated and ignored by Git)

Chunks never cross PDF page boundaries. Each retains document name, PDF page, printed page when detected, section path, bounding boxes, extraction method, content type, and safety tags.

## Local setup (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
maintenance-ingest
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
- The selected result expands to the immediately adjacent chunks on the same page, improving split procedures and tables without crossing page boundaries; each original chunk remains separately validated and cited.
- Low-confidence retrieval returns an explicit insufficient-evidence response.
- Safety-tagged evidence is surfaced and safety wording is not rewritten into new procedures.

## Known limitations

- OCR fallback requires a local Tesseract installation; low-text pages are recorded as warnings when it is unavailable.
- Evidence highlighting uses chunk block bounding boxes, not word-perfect character coordinates.
- The extractive answerer is safer and auditable but less conversational than a carefully constrained LLM.
- Qdrant indexing is CPU-heavy on first run because embeddings are created locally.
- The evidence selector is tuned on an initial five-question corpus; its perfect answer-level snapshot is a regression baseline, not evidence of generalization.
- Diagram interpretation is not implemented yet; diagrams are currently retrievable only through nearby text and captions.
- PostgreSQL stores query audit events, while document metadata remains in versioned JSONL artifacts for this MVP.

## Recommended next upgrades

1. Expand and manually verify the golden set, including adversarial and unanswerable questions; tune the abstention threshold from measured precision/coverage.
2. Add reranking after hybrid retrieval and compare it against the existing baseline before accepting the added latency.
3. Add OCR layout regression tests and install Tesseract for scanned manuals.
4. Add table-specific normalization and diagram/callout extraction only after text retrieval metrics are stable.
5. Add an optional citation-constrained LLM synthesis layer with sentence-level entailment checks; retain extractive fallback.
6. Pin the Qdrant container image after testing a deployment target, add authentication/TLS, migrations, observability, rate limits, and background indexing.
7. Consider a knowledge graph only if cross-document evaluation demonstrates a measurable failure that graph traversal fixes.
