# Resume and interview material

## Project title

Multimodal Maintenance Intelligence Agent — grounded RAG for industrial troubleshooting

## Verified resume bullets

- Built a grounded industrial-document assistant over 3 ABB ACS580 manuals and 1,482 pages, producing 11,888 page-bounded chunks with stable provenance, safety tags, table classification, and PDF bounding boxes.
- Implemented and evaluated term-frequency, BM25, BGE-small dense, and weighted reciprocal-rank hybrid retrieval; measured 100% Recall@10 for BM25 and lexical-heavy hybrid versus 50% for dense retrieval on a manually verified 12-question answerable set.
- Achieved 100% acceptable-page answer citation accuracy across 12 answerable questions and 100% abstention recall with a 0% false-answer rate across 8 adversarial unsupported questions using adjacent-chunk expansion, technical-code boosting, document-aware evidence selection, and conjunctive safety gates.
- Developed a FastAPI service and Next.js/TypeScript interface with citation validation, highlighted PDF evidence, optional PostgreSQL audit logging, Qdrant vector search, request IDs, structured logs, readiness checks, Prometheus metrics, and optional API-key protection.
- Added a passing Python test suite, frontend lint and production-build gates, Docker Compose orchestration, container health checks, and GitHub Actions CI.
- Built a provenance-preserving visual inventory of 1,462 page-linked candidates (372 raster images and 1,090 vector regions), with stable IDs, bounding boxes, and nearby captions, as the measurable foundation for later diagram understanding.
- Created a deterministic visual-evaluation workflow spanning all six document/type strata; manually inspected 24 candidates and observed 100% meaningful-candidate precision with an 86.2% Wilson 95% lower bound, while explicitly excluding recall and comprehension claims.

Every metric above must remain qualified as a regression baseline because the manually verified set contains only 20 questions from one product family.

## Thirty-second explanation

I built a grounded troubleshooting assistant for ABB ACS580 drives. It verifies manual revisions by checksum, extracts page-bounded text and table chunks with source coordinates, inventories page-linked visual candidates, combines BM25 and Qdrant dense retrieval, and returns extractive answers with validated page citations and highlighted evidence. I evaluated retrieval and abstention instead of assuming embeddings helped; BM25 beat the dense model, so the deployed fusion is intentionally lexical-heavy. The API includes tests, observability, security headers, audit logging, Docker orchestration, and a Next.js evaluation dashboard.

## Best demo sequence

1. Show the evaluation snapshot and disclose the 20-question, single-product-family limitation.
2. Ask what warning A2B1 means and open the highlighted evidence page.
3. Ask for the connected motor's bearing replacement torque and show the explicit abstention.
4. Open `/ready` and `/metrics` to demonstrate production engineering.
5. Show the golden CSV and explain how evaluation changed the architecture.
6. Show `visual_labels.csv`, explain stratified sampling, and distinguish candidate precision from diagram understanding.

## Claims to avoid

- Do not call the application production-ready or safety-certified.
- Do not claim dense retrieval improved results.
- Do not claim visual diagram understanding or complete OCR.
- Describe the visual records as detector candidates until a manually labeled visual evaluation set exists.
- Do not generalize the 20-question, single-family metrics to unseen equipment.
- Do not claim Docker runtime verification on the current computer.
