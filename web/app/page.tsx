"use client";

import { FormEvent, useEffect, useState } from "react";

type Citation = {
  chunk_id: string;
  source_filename: string;
  pdf_page: number;
  printed_page: string;
  evidence: string;
  bboxes: number[][];
};

type Answer = {
  answer: string;
  abstained: boolean;
  confidence: number;
  citations: Citation[];
};

type Evaluation = {
  corpus: { documents: number; pages: number; chunks: number; vectors: number };
  golden_set: { answerable_questions: number; unanswerable_questions: number; warning: string };
  retrieval: { bm25_recall_at_10: number; dense_recall_at_10: number; hybrid_recall_at_10: number; hybrid_mrr: number };
  answering: { answerable_citation_accuracy: number; unanswerable_abstention_recall: number };
};

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("What does warning A2B1 mean and what should I check?");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);

  useEffect(() => {
    fetch(`${API}/evaluation`)
      .then((response) => (response.ok ? response.json() : null))
      .then(setEvaluation)
      .catch(() => setEvaluation(null));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setAnswer(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <div className="eyebrow">ABB ACS580 · grounded document intelligence</div>
        <h1>Maintenance Intelligence</h1>
        <p>Ask a troubleshooting question. Every response must be traceable to the locked manual corpus.</p>
      </header>

      {evaluation && (
        <section className="evaluation" aria-label="Verified evaluation snapshot">
          <div className="evaluation-heading">
            <div>
              <div className="eyebrow">Measured system evidence</div>
              <h2>Evaluation snapshot</h2>
            </div>
            <span>{evaluation.golden_set.answerable_questions + evaluation.golden_set.unanswerable_questions} verified questions</span>
          </div>
          <div className="metric-grid">
            <div><strong>{evaluation.corpus.chunks.toLocaleString()}</strong><span>citable chunks</span></div>
            <div><strong>{Math.round(evaluation.retrieval.hybrid_recall_at_10 * 100)}%</strong><span>hybrid Recall@10</span></div>
            <div><strong>{Math.round(evaluation.answering.answerable_citation_accuracy * 100)}%</strong><span>expected-page citation</span></div>
            <div><strong>{Math.round(evaluation.answering.unanswerable_abstention_recall * 100)}%</strong><span>abstention recall</span></div>
          </div>
          <p className="evaluation-warning">{evaluation.golden_set.warning}</p>
        </section>
      )}

      <form onSubmit={submit}>
        <label htmlFor="question">Technician question</label>
        <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} rows={4} />
        <button disabled={loading || question.trim().length < 2}>{loading ? "Searching evidence…" : "Find grounded answer"}</button>
      </form>

      {error && <div className="error">{error}</div>}
      {answer && (
        <section className={answer.abstained ? "answer abstained" : "answer"}>
          <div className="status-row">
            <span>{answer.abstained ? "Insufficient evidence" : "Evidence-backed response"}</span>
            <span>Confidence {Math.round(answer.confidence * 100)}%</span>
          </div>
          <h2>Answer</h2>
          <p>{answer.answer}</p>
          <h2>Supporting evidence</h2>
          <div className="evidence-grid">
            {answer.citations.map((citation) => (
              <article key={citation.chunk_id}>
                <div className="citation">{citation.source_filename} · page {citation.printed_page}</div>
                <blockquote>{citation.evidence}</blockquote>
                <a href={`${API}/evidence/${encodeURIComponent(citation.chunk_id)}/image`} target="_blank">
                  Open highlighted page evidence
                </a>
              </article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
