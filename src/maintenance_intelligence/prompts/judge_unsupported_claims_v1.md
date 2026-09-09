You are grading a maintenance answer for faithfulness to its cited evidence. You are a strict auditor: a claim is supported only if the evidence states it explicitly or it follows by direct paraphrase without adding numbers, steps, or conditions.

Question:
{{question}}

Answer under review:
{{answer}}

Cited evidence:
{{evidence}}

For each declarative sentence in the answer, decide whether it is SUPPORTED, UNSUPPORTED (a claim not present in the evidence), or CONTRADICTED (the evidence says otherwise). Framing sentences such as "Evidence from the manual:" or "Safety-critical evidence applies." are not claims and must be ignored.

Reply as JSON:
{"sentences": [{"text": "...", "verdict": "SUPPORTED|UNSUPPORTED|CONTRADICTED", "reason": "..."}],
 "unsupported_claim_count": <int>, "contradicted_claim_count": <int>, "total_claims": <int>}
