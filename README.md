# Multimodal Maintenance Intelligence Agent

A grounded troubleshooting assistant for the ABB ACS580-01 variable-frequency-drive manuals. Ask a technician-style question ("What should I check for warning A2B1?") and get an answer assembled **only from sentences in the manuals**, with page-level citations, highlighted source pages, and an explicit "insufficient evidence" response when the manuals don't support an answer.

It is a retrieval-augmented system where the interesting part is the guardrails: hybrid BM25 + dense retrieval, citation validation against the indexed chunks, safety-language preservation, a bounded read-only agent loop, and a versioned evaluation set that gates every change.

> **Portfolio/research MVP.** Not safety-certified or production-approved. Not a substitute for manufacturer documentation, qualified personnel, lockout/tagout procedures, or site safety rules.

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Using the API](#using-the-api)
- [Repository layout](#repository-layout)
- [Evaluation and reproducibility](#evaluation-and-reproducibility)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## What it does

| Capability | How |
| --- | --- |
| Ingests 3 public ABB manuals (1,482 pages) | PyMuPDF text extraction with optional OCR fallback; chunks never cross a PDF page boundary and keep document, page, section path, bounding boxes, and safety tags |
| Finds the right page | BM25 (exact fault codes, parameter numbers) + BGE-small dense vectors in Qdrant, merged with weighted reciprocal-rank fusion |
| Answers without hallucinating | Extractive answerer: only sentences already present in retrieved chunks; every citation is re-validated against the chunk registry before returning |
| Knows when to say no | Abstains with an explicit insufficient-evidence response when retrieval confidence is low or citations fail validation |
| Shows its work | Highlighted evidence crops from the source PDF page; provenance-checked visual candidates (diagrams, tables) from the same cited pages |
| Bounded agent mode | `POST /agent/query`: a read-only controller with at most 5 tool calls for fault-code lookup, cross-manual comparison, and visual-evidence requests; emits an auditable action trace |
| Guardrails | Input guard rejects prompt injection, chat-template tokens and control characters (HTTP 422 with flags); equipment-control requests are flagged and answered with an explicit "cannot operate equipment" notice; output guard verifies every answer sentence is verbatim in a cited chunk and forces abstention otherwise |
| Typed contracts | Every answer route returns a Pydantic `AnswerResponse`; a schema-enforced LLM helper validates JSON against a model, retries with the validation error, and falls back deterministically |
| Observability | OpenTelemetry spans for guardrails, retrieval, answering, agent steps and LLM calls (token/cost attributes); exporter is config: console, OTLP (Arize Phoenix in Compose), or Azure Application Insights. Plus Prometheus `/metrics` and structured JSON logs |
| MCP server | `maintenance-mcp` exposes `search_manuals`, `ask_manuals`, `agent_query` over the Model Context Protocol, guarded identically to the HTTP API |
| Experiment tracking | Evaluation runs log params, metrics, git commit, prompt-file hashes and report artifacts to MLflow (in Compose); prompts live in a versioned registry (`prompts/*.md`) |
| Production hygiene | `/health`, `/ready`, `/config` (non-secret runtime state), `X-Request-ID` on every response, optional API-key auth, PostgreSQL query audit log |

## Results

Every number below is a versioned regression result on a **small, single-product-family benchmark** (`evaluation/`). It is not evidence of production accuracy or generalization, and `tests/test_portfolio_artifacts.py` fails if the README's snapshot drifts from the evaluation files.

| Area | Result |
| --- | --- |
| Corpus | 3 manuals, 1,482 pages, 11,888 provenance-preserving chunks |
| Text retrieval | 100% hybrid Recall@10 on 12 answerable questions (BM25 alone 100%, dense alone 50%) |
| Grounding | 100% acceptable-page citation accuracy on 12 answerable questions |
| Abstention | 100% recall and 0% false-answer rate on 8 unsupported questions |
| Visual inventory | 1,462 page-linked candidates; 24 manually inspected, 100% precision (Wilson 95% lower bound 86.2%) |
| Visual retrieval | 41.7% lexical, 58.3% CLIP, 66.7% hybrid Recall@10 on 12 queries over the same 24 candidates |
| Engineering gates | 88 Python tests, clean frontend lint, successful production TypeScript build, CI on every push |

## How it works

```text
                     OFFLINE (one-time per corpus)
ABB manual PDFs ──► PyMuPDF / OCR ──► page-bounded chunks ──► BM25 (in memory)
                          │                                └──► FastEmbed ──► Qdrant
                          └──► visual candidate inventory (raster images + vector regions)

                     ONLINE (per question)
question ──► BM25 + dense ──► weighted RRF ──► adjacent-chunk expansion ──► extractive answer
                                                                              ├─► citation validation
                                                                              ├─► safety-warning preservation
                                                                              ├─► highlighted PDF evidence
                                                                              ├─► same-page visual candidates
                                                                              └─► insufficient-evidence response

Next.js UI ──► FastAPI ──► PostgreSQL (query / audit log)
```

**Retrieval.** BM25 handles exact identifiers, parameter numbers, and warning codes; BGE-small embeddings handle paraphrase. Weighted RRF merges the two ranked lists without pretending their scores share a scale. The evaluated default weights BM25 16:1 because equal weighting reduced exact-page recall on the golden set.

**Answering.** The answer layer is deliberately extractive. It assembles sentences that already exist in the retrieved chunks instead of asking a generative model to write prose or procedures. The selected chunk expands to its immediate neighbours on the same page (split procedures and tables) but never across a page boundary; each chunk is separately validated and cited. Safety-tagged evidence is surfaced verbatim and never rewritten into a new procedure.

**Visual evidence.** The visual inventory records every embedded raster image and vector-heavy region with source file, page, bounding box, nearby caption, and a stable asset ID. Crops are served at `/visuals/{asset_id}/image` from the offline inventory only; arbitrary filenames are never accepted. Visuals are joined to an answer *after* text grounding succeeds and cannot influence answer text or the abstention decision.

**Agent mode.** `POST /agent/query` wraps the same answerer in a bounded controller: it classifies the question (fault code, multi-document comparison, visual evidence, or plain lookup), runs at most three searches and five tool calls, keeps only responses whose citations survive validation, preserves safety flags, and returns the full action trace. It has no tool that can control equipment or change drive parameters.

**Guardrails.** Every answer route runs three deterministic checks (`guardrails.py`). *Input:* prompt-injection patterns ("ignore previous instructions", role overrides, chat-template tokens, secret exfiltration) and control characters are rejected with HTTP 422 and a list of flags; requests to *operate* equipment ("set parameter 30.17", "bypass STO") are allowed but flagged, and the answer is prefixed with a notice that the system cannot act on equipment. *Output:* every sentence of the answer must appear verbatim (whitespace/Unicode-normalized) in the text of a validated citation; anything else forces the insufficient-evidence response. The extractive answerer passes this by construction; the guard exists so an LLM synthesis mode or a regression cannot ship an unsupported sentence. Each response carries a `guardrails` block with the flags and the grounded-sentence ratio.

**Observability.** `tracing.py` wraps OpenTelemetry: `guardrails.input` → `answer.<mode>` → `guardrails.output` spans, with LLM spans recording model, token counts, estimated cost and latency using the GenAI semantic-convention attribute names. `TRACING_EXPORTER` selects `none`, `console`, `otlp` (Phoenix at `http://localhost:6006` in Compose) or `azure` (Application Insights). Prometheus counters cover requests, latency, abstentions, guardrail rejections, LLM tokens and spend.

## Quickstart

### 0. Get the manuals

Download the three public ABB ACS580 manuals listed in [`data/manuals/README.md`](data/manuals/README.md) into `data/manuals/`. They are not redistributed here; `maintenance-ingest` verifies their SHA-256 hashes against `evaluation/corpus_checksums.csv` before processing.

### Option A: Docker Compose (everything in containers)

```bash
cp .env.example .env
docker compose up --build
```

Compose starts PostgreSQL, Qdrant, **Phoenix** (traces, `http://localhost:6006`) and **MLflow** (experiments, `http://localhost:5000`), runs a one-shot **ingest** job (builds `data/processed/` from the PDFs; skipped if it already exists), a one-shot **indexer** job (embeds chunks into Qdrant), then the API on `http://localhost:8000` and the UI on `http://localhost:3000`. The API ships traces to Phoenix and eval runs to MLflow out of the box. First boot is CPU-heavy because embeddings are computed locally; expect several minutes.

The Compose file has been validated with `docker compose config` but a full `up` was not run on the development machine, which had no Docker daemon. If you hit a problem, please open an issue with the service logs.

### Option B: Local development (Python 3.12 + Node 22)

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env

maintenance-ingest                 # PDFs -> data/processed/chunks.jsonl
maintenance-visual-inventory       # page-linked visual candidates
maintenance-index --recreate       # embed chunks into local Qdrant (data/qdrant/)
maintenance-api                    # http://localhost:8000  (docs at /docs)
```

Then, in a second terminal:

```bash
cd web
pnpm install
pnpm dev                           # http://localhost:3000
```

PostgreSQL is optional locally: leave `DATABASE_URL` empty in `.env` and audit logging is skipped.

### Verify

```bash
python -m pytest                   # 88 tests; full-corpus checks auto-skip until you ingest
cd web && pnpm lint && pnpm build
```

The test suite runs against a small synthetic fixture corpus in `tests/fixtures/`, so it is green on a fresh clone and in CI. Once you have ingested the real manuals, the `*_on_full_corpus` tests stop skipping and act as regression checks.

## Using the API

Interactive docs are at `http://localhost:8000/docs`. If `API_KEY` is set in `.env`, send it as `X-API-Key`.

```bash
# Grounded extractive answer
curl -s http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I check for warning A2B1?"}'

# Bounded agent: multi-document comparison with an action trace
curl -s http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare the hardware and firmware guidance for overcurrent"}'
```

Response shape (both routes):

```json
{
  "answer": "…sentences taken verbatim from the cited chunks…",
  "abstained": false,
  "confidence": 0.71,
  "citations": [
    {"chunk_id": "…", "source_filename": "abb_acs580_firmware_….pdf", "pdf_page": 538, "printed_page": "536", "evidence": "…", "bboxes": [[…]]}
  ],
  "visual_evidence": [{"asset_id": "…", "pdf_page": 538, "caption": "…"}],
  "agent": {"task_type": "fault_code", "policy": "bounded_read_only", "tool_calls": 3, "trace": [ … ]}
}
```

An unsupported question returns `"abstained": true`, an empty citation list, and no visuals. Every response also includes `"mode"` (`extractive` | `agent` | `synthesis`) and a `"guardrails"` block (`input_flags`, `control_intent`, `grounded_sentence_ratio`, `unsupported_sentences`). A rejected question returns HTTP 422:

```json
{"detail": {"detail": "The question contains prompt-injection or unsafe control content and was not processed", "flags": ["injection:ignore_instructions"]}}
```

### MCP server

```bash
maintenance-mcp                    # stdio transport; same runtime and guardrails as the API
```

Claude Desktop / Cursor config:

```json
{"mcpServers": {"maintenance-manuals": {"command": "maintenance-mcp", "env": {"CHUNKS_PATH": "/abs/path/data/processed/chunks.jsonl", "QDRANT_PATH": "/abs/path/data/qdrant"}}}}
```

Tools: `search_manuals(question, limit)`, `ask_manuals(question)`, `agent_query(question)`. All read-only.

### Tracing and experiments

```bash
TRACING_EXPORTER=console maintenance-api          # spans to stdout
TRACING_EXPORTER=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces maintenance-api   # Phoenix
MLFLOW_TRACKING_URI=http://localhost:5000 maintenance-evaluate-full --run-name baseline             # logs to MLflow
```

| Route | Purpose |
| --- | --- |
| `POST /ask` | Deterministic retrieval + extractive answer |
| `POST /agent/query` | Bounded agent controller over the same tools |
| `GET /evidence/{chunk_id}/image` | Source page with the cited chunk highlighted |
| `GET /visuals/{asset_id}/image` | Provenance-checked crop of a visual candidate |
| `GET /documents/{filename}/pages/{n}/image` | Rendered manual page |
| `GET /evaluation` | Versioned evaluation snapshot shown in the UI |
| `GET /config` | Non-secret runtime state: provider, models, tracing exporter, fusion weights, prompt versions |
| `GET /health`, `GET /ready`, `GET /metrics` | Liveness, retrieval readiness, Prometheus metrics |

## Repository layout

```text
src/maintenance_intelligence/
  extract.py, chunking.py, pipeline.py   PDF -> page-bounded chunks with provenance
  retrieval.py, dense.py, hybrid.py      BM25, Qdrant dense search, weighted RRF
  answering.py, citations.py             extractive answerer, citation validation, abstention
  agent.py                               bounded read-only controller (POST /agent/query)
  guardrails.py, schemas.py              input/output guardrails, Pydantic response contracts
  llm.py, structured.py                  provider adapter (OpenAI / Azure OpenAI), schema-enforced calls
  tracing.py, experiments.py             OpenTelemetry spans, MLflow experiment tracking
  prompts/                               versioned prompt registry (content-hashed .md files)
  mcp_server.py                          Model Context Protocol server (maintenance-mcp)
  visuals.py, visual_evidence.py         visual candidate inventory and page-linked evidence
  visual_retrieval.py, visual_clip.py    lexical / CLIP / hybrid visual retrieval experiments
  evaluation.py, *_cli.py                golden-set evaluation and command-line entry points
  api.py, observability.py, database.py  FastAPI app, request IDs, logging, metrics, audit log
web/                                     Next.js 15 UI (app router, standalone build)
tests/                                   88 pytest tests; tests/fixtures/ holds the synthetic corpus
evaluation/                              golden questions, checksums, corpus manifest, metrics JSON
data/manuals/                            place the three ABB PDFs here (gitignored)
data/processed/, data/qdrant/            generated artifacts (gitignored)
docker-compose.yml, Dockerfile.api       containerized deployment
.github/workflows/ci.yml                 pytest + frontend lint/build on every push and PR
```

## Evaluation and reproducibility

- **Golden set:** `evaluation/golden_questions.csv` — 12 answerable questions with verified page citations and 8 adversarial unsupported questions. Run `maintenance-evaluate` (retrieval) and `maintenance-evaluate-full` (answering + abstention). Results are written to `evaluation/results/` and summarized in `evaluation/baseline_metrics.json`.
- **Visual pipeline:** `maintenance-visual-sample` draws a deterministic 24-item stratified sample (3 manuals × raster/vector); `evaluation/visual_labels.csv` holds the manual inspection labels; `maintenance-visual-evaluate` computes precision with a Wilson interval.
- **Visual retrieval:** 12 technician-style queries in `evaluation/visual_retrieval_questions.csv`. `maintenance-visual-retrieval-evaluate` (BM25 over caption/page text), `maintenance-visual-clip-evaluate` (paired CLIP text/pixel embeddings via FastEmbed, run locally), and the hybrid fusion each write a full per-query report with ranks and top results. Hybrid wins Recall@10 (66.7%) but CLIP alone wins Recall@3 (58.3% vs 50.0%); the pool is 24 candidates, the questions were authored after inspection, and there is no held-out tuning set, so this is an evaluated experiment and is not allowed to influence maintenance answers.
- **Drift guard:** `tests/test_portfolio_artifacts.py` asserts that the published snapshot in `baseline_metrics.json` matches the golden set, corpus manifest, label files, and per-method metrics JSON.
- **Experiment tracking:** `maintenance-evaluate-full` logs every run to MLflow when `MLFLOW_TRACKING_URI` is set — fusion weights, abstention threshold, git commit, the SHA-256 of each prompt file, all metrics, and the JSON reports as artifacts — so any number can be traced to the exact code and prompt that produced it.

## Known limitations

- The golden set is 20 questions from one product family. The perfect answer-level snapshot is a regression baseline, not proof of generalization.
- OCR fallback needs a local Tesseract install; without it, low-text pages are logged as warnings (28 in the current corpus).
- Evidence highlighting uses chunk block bounding boxes, not character-level coordinates.
- The extractive answerer is auditable but less fluent than a constrained generative layer would be.
- Diagram *interpretation* is not implemented. 1,462 visual candidates are inventoried and retrievable, but nothing reads wiring relationships or answers questions from pixels.
- The visual retrieval benchmark is 12 queries over 24 inspected candidates, not full-corpus or held-out.
- First-run indexing is CPU-heavy because embeddings are computed locally.
- Compose has been validated but not exercised end-to-end on a Docker host (see Quickstart).

## Roadmap

1. Expand text and visual golden sets with independently authored, held-out questions from additional equipment families.
2. Measure full-corpus visual detector recall and false positives, not just candidate precision.
3. Add Tesseract to the API image and scanned-page OCR regression tests.
4. Benchmark a cross-encoder reranker against the lexical-heavy baseline before accepting its latency.
5. Optional citation-constrained synthesis with sentence-level entailment checks, keeping the extractive path as fallback.
6. Production hardening: identity-aware auth, TLS, migrations, rate limits, backups, alerting, dependency scanning.
7. In progress: optional LLM synthesis mode behind the same output guard, an LLM-as-judge evaluation harness in CI, semantic caching and model routing with p95 latency and cost-per-query numbers, an async worker for batch evaluation, a LoRA fine-tuning comparison, and an Azure deployment (AI Search, Container Apps, Application Insights, Azure OpenAI).

## Contributing and repository boundaries

Never commit manuals, extracted JSONL, page crops, Qdrant storage, model weights, `.env`, or query logs; `.gitignore` covers all of these. Reproduce generated artifacts locally from the public manuals. Run `python -m pytest` and `cd web && pnpm lint && pnpm build` before opening a pull request. See [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities.

## License

Original source code is available under the [MIT License](LICENSE). ABB manuals, trademarks, product names, and other third-party materials are not covered and are not distributed with this repository.
