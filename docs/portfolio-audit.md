# Portfolio completion audit

This audit maps each portfolio competency to current repository evidence. “Verified” means a local test, build, generated report, or inspected artifact supports the claim. It does not mean the MVP is production-approved.

| Competency | Evidence | Status | Honest boundary |
| --- | --- | --- | --- |
| Applied AI / RAG | Page-bounded ingestion, BM25, BGE-small dense retrieval, weighted RRF, extractive answering, citation validation | Verified locally | One ABB drive family; extractive rather than generative synthesis |
| ML evaluation | 20-question grounded-answer set; term-frequency/BM25/dense/hybrid reports; answer and abstention metrics | Verified locally | Small manually authored regression set, not evidence of generalization |
| Multimodal evaluation | 1,462 visual candidates; 24 inspected labels; 12 visual queries; lexical, CLIP, and hybrid reports | Verified locally | Candidate-pool benchmark only; no full-corpus visual recall or diagram comprehension |
| Python backend | FastAPI routes, configuration, Qdrant integration, PostgreSQL audit repository, typed dataclasses | Verified by tests | Database and vector-service integration were not exercised through Docker on this machine |
| Full-stack engineering | Next.js/TypeScript UI for questions, evaluation metrics, citations, highlighted pages, and visual cards | Lint and production build verified | Interactive browser QA blocked by localhost URL policy |
| Testing | 38 Python tests plus frontend lint and production TypeScript build | Verified locally | No browser end-to-end suite and no live Docker integration test |
| CI | GitHub Actions runs Python 3.12 tests, frontend lint, and frontend production build | Configuration inspected | Workflow has not been observed on GitHub because no remote run was requested |
| Deployment | Multi-stage API/UI Dockerfiles, Compose orchestration, health checks, persistent volumes | Source inspected | Docker CLI/runtime unavailable locally; do not claim a successful Compose launch |
| Observability | Liveness, readiness, Prometheus request/latency/answer counters, request IDs, structured logs | Verified by tests | No deployed collector, dashboard, alerting, or retention policy |
| Security | Optional constant-time API-key check, CORS allowlist, traversal checks, security headers, no-store answers, secret exclusions | Verified by tests/source | Not complete authentication; no TLS, RBAC, rate limiting, or production threat assessment |
| Safety grounding | Safety tags, preserved warning text, conjunctive gates, explicit insufficient-evidence response | Verified on the regression set | Not safety-certified and must not control equipment or replace manufacturer/site procedures |
| Documentation | README, security policy, evaluation artifacts, resume/interview guide, this audit | Inspected | Claims must retain their sample-size and environment qualifications |

## Reproduce the verified gates

```powershell
python -m pytest
cd web
pnpm lint
pnpm build
```

The multimodal benchmark additionally requires the ignored manual corpus and local CLIP model cache:

```powershell
maintenance-visual-inventory
maintenance-visual-sample
maintenance-visual-evaluate
maintenance-visual-retrieval-evaluate
maintenance-visual-clip-evaluate
```

## Remaining work before any production claim

Run Compose on a Docker-enabled host; expand independently authored and held-out text/visual evaluations; measure full-corpus visual detector recall; add browser end-to-end and live service integration tests; add real identity, TLS, rate limits, secret management, dependency scanning, backup/restore tests, alerting, and an industrial security and safety review.
