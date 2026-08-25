# Multimodal Maintenance Intelligence Agent

A grounded, text-first troubleshooting assistant for the ABB ACS580-01 variable-frequency-drive family. It ingests three public manuals, preserves page-level provenance and safety language, combines BM25 and dense retrieval, answers only from retrieved evidence, validates citations, renders highlighted source pages, and abstains when support is weak.

> **Portfolio/research MVP:** This project is not safety-certified, production-approved, or a substitute for manufacturer documentation, qualified personnel, lockout/tagout procedures, or site safety rules.

## Verified project snapshot

| Area | Verified result |
| --- | --- |
| Corpus | 3 manuals, 1,482 pages, 11,888 provenance-preserving chunks |
| Text retrieval | 100% hybrid Recall@10 on 12 answerable regression questions |
| Grounding | 100% acceptable-page citation accuracy on 12 answerable questions |
| Abstention | 100% recall and 0% false-answer rate on 8 unsupported questions |
| Visual inventory | 1,462 page-linked candidates; 24 manually inspected labels |
| Visual retrieval | 41.7% lexical, 58.3% CLIP, and 66.7% hybrid Recall@10 on 12 queries over the same 24 candidates |
| Engineering gates | 38 Python tests, clean frontend lint, successful production TypeScript build |

Every percentage above is a versioned regression result from a small, single-product-family benchmark. It is not evidence of production accuracy or generalization.

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
- Versioned corpus summary: `evaluation/corpus_manifest.json`
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
maintenance-visual-retrieval-evaluate
maintenance-visual-clip-evaluate
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

The Docker definitions are included for reproducible deployment, but a Compose launch was not verified on the development machine because Docker was unavailable there.

## Repository boundaries

The public repository intentionally excludes ABB PDFs, extracted JSONL data, page crops, Qdrant storage, model weights, virtual environments, dependency folders, build output, secrets, and query logs. Contributors must download the public manuals themselves and reproduce generated artifacts locally. Never commit production manuals, equipment identifiers, credentials, or technician query logs.

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

## Multimodal retrieval experiment

Twelve technician-style visual queries were manually verified against the inspected candidate set. On the identical 24-candidate pool, text-linked BM25 achieved 41.7% asset Recall@10 and 0.333 MRR; paired CLIP text/pixel embeddings achieved 58.3% Recall@10 and 0.403 MRR; untuned 1:1 reciprocal-rank fusion achieved the best Recall@10 at 66.7% but lower Recall@3 than CLIP (50.0% versus 58.3%). This demonstrates complementary text and pixel signals without pretending the current sample establishes generalization.

The CLIP models run locally through FastEmbed, and their cache is excluded from Git. The benchmark reports preserve every query, target asset, rank, and top result in `evaluation/visual_retrieval_metrics.json`, `evaluation/visual_clip_metrics.json`, and `evaluation/visual_hybrid_metrics.json`. The questions were authored after visual inspection, the pool contains only 24 candidates, and there is no held-out tuning set; therefore visual retrieval remains an evaluated experiment and is not allowed to influence maintenance answers.

## Known limitations

- OCR fallback requires a local Tesseract installation; low-text pages are recorded as warnings when it is unavailable.
- Evidence highlighting uses chunk block bounding boxes, not word-perfect character coordinates.
- The extractive answerer is safer and auditable but less conversational than a carefully constrained LLM.
- Qdrant indexing is CPU-heavy on first run because embeddings are created locally.
- The evidence selector is evaluated on only 20 questions from one product family; its perfect answer-level snapshot is a regression baseline, not evidence of generalization.
- Diagram interpretation is not implemented yet; 1,462 page-linked visual candidates have been inventoried, but they are not yet semantically interpreted or used for answering.
- The multimodal retrieval benchmark contains only 12 queries over 24 inspected candidates and is not a full-corpus or held-out evaluation.
- PostgreSQL stores query audit events, while document metadata remains in versioned JSONL artifacts for this MVP.

## Recommended next upgrades

1. Expand the text and visual golden sets with independently authored, held-out questions from additional equipment families.
2. Measure full-corpus visual detector recall and false positives rather than candidate precision alone.
3. Install Tesseract and add scanned-page OCR layout regression tests.
4. Compare a cross-encoder reranker against the existing lexical-heavy text baseline before accepting its latency.
5. Add optional citation-constrained synthesis with sentence-level entailment checks while retaining the extractive fallback.
6. Verify Compose on a Docker-enabled host, then add identity-aware authentication, TLS, migrations, rate limits, backups, alerting, and dependency scanning.
7. Consider a knowledge graph only if cross-document evaluation demonstrates a measurable failure that graph traversal fixes.

## Publishing notes

A clean clone requires the three public manuals listed in `data/manuals/README.md`. Copy `.env.example` to `.env`, keep all secrets local, and run the verification commands before opening a pull request. No license has been selected; choose one deliberately before granting reuse rights.
