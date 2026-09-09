"""Model Context Protocol server exposing the grounded retrieval tools.

Any MCP-capable client (Claude Desktop, Cursor, an agent framework) can call:

* ``search_manuals`` – ranked, page-deduplicated evidence chunks
* ``ask_manuals``    – grounded extractive answer with validated citations
* ``agent_query``    – the bounded read-only controller with its action trace

The server is read-only by construction: there is no tool that writes to the
corpus, the database, or any equipment. Guardrails run on every call exactly as
they do for the HTTP API.

Run with ``maintenance-mcp`` (stdio transport) after ``maintenance-index``.
"""

from __future__ import annotations

from typing import Any

from maintenance_intelligence.answering import INSUFFICIENT
from maintenance_intelligence.guardrails import apply_output_guard, check_input


def create_mcp_server(runtime: Any, *, name: str = "maintenance-intelligence"):
    """Build an MCP server bound to an :class:`AppRuntime`-like object.

    ``runtime`` needs ``answerer``, ``agent``, ``chunks_by_id``, ``visual_index``
    and ``answerer.hybrid`` (a retriever with ``search(query, limit)``).
    """
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name,
        instructions=(
            "Read-only retrieval over ABB ACS580 maintenance manuals. Answers are "
            "extractive and cited to PDF pages. An 'abstained' result means the "
            "manuals do not support an answer; do not guess on the user's behalf."
        ),
    )

    def _guarded(question: str, produce) -> dict[str, Any]:
        decision = check_input(question)
        if not decision.allowed:
            return {
                "answer": INSUFFICIENT,
                "abstained": True,
                "confidence": 0.0,
                "citations": [],
                "visual_evidence": [],
                "guardrails": {"input_flags": decision.flags, "rejected": True, "reason": decision.reason},
            }
        response = produce(question)
        return apply_output_guard(
            response,
            runtime.chunks_by_id,
            input_decision=decision,
            insufficient_text=INSUFFICIENT,
        )

    @server.tool()
    def search_manuals(question: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the top evidence chunks for a maintenance question, one per PDF page."""
        limit = max(1, min(int(limit), 10))
        decision = check_input(question)
        if not decision.allowed:
            return []
        hits = runtime.answerer.hybrid.search(question, limit=limit)
        results = []
        for item in hits:
            chunk = item.result.chunk
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_filename": chunk.source_filename,
                    "pdf_page": chunk.pdf_page,
                    "printed_page": chunk.printed_page,
                    "section_path": list(getattr(chunk, "section_path", [])),
                    "score": round(float(item.rrf_score), 6),
                    "contains_safety_warning": bool(chunk.contains_safety_warning),
                    "text": chunk.text[:1200],
                }
            )
        return results

    @server.tool()
    def ask_manuals(question: str) -> dict[str, Any]:
        """Grounded extractive answer with validated page citations, or an abstention."""

        def produce(text: str) -> dict[str, Any]:
            response = runtime.answerer.answer(text)
            response["visual_evidence"] = (
                runtime.visual_index.for_citations(response.get("citations", []))
                if not response.get("abstained") and runtime.visual_index is not None
                else []
            )
            return response

        return _guarded(question, produce)

    @server.tool()
    def agent_query(question: str) -> dict[str, Any]:
        """Bounded read-only agent: fault-code lookup, cross-manual comparison, visual evidence."""
        return _guarded(question, runtime.agent.run)

    return server


def main() -> None:
    from maintenance_intelligence.api import build_runtime
    from maintenance_intelligence.config import Settings

    runtime = build_runtime(Settings())
    try:
        create_mcp_server(runtime).run(transport="stdio")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
