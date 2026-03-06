"""Reasoning Agent."""

import json
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()

SYSTEM_PROMPT = """Synthesize answer from evidence. Rules: never invent numbers, cite (Page X) for every claim, use math results exactly.
Return ONLY JSON: {"answer":"text with (Page X)","citations":[{"page":int,"text":"min 10 words"}],"confidence":0.0-1.0,"data_used":["points"]}"""


class ReasoningAgent:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id
        from agents.gemini_client import create_model
        self.model = create_model(system_instruction=SYSTEM_PROMPT, temperature=0.2, max_output_tokens=2048)

    def run(self, query: str, evidence: List[Dict]) -> Dict:
        from agents.gemini_client import generate_with_retry
        ev_text = self._format(evidence)
        try:
            raw = generate_with_retry(self.model, f"Query: {query}\n\nEvidence:\n{ev_text}\n\nSynthesize answer.", trace_id=self.trace_id)
            clean = raw.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
            result = json.loads(clean)
            result.setdefault("answer", clean)
            result.setdefault("citations", self._auto_citations(evidence))
            result.setdefault("confidence", self._conf(evidence))
            result.setdefault("data_used", [])
            result["citations"] = [c for c in result["citations"] if isinstance(c, dict) and len(c.get("text", "")) >= 10]
            if not result["citations"]:
                result["citations"] = self._auto_citations(evidence)
            return result
        except json.JSONDecodeError:
            return {"answer": raw if "raw" in dir() else "Error", "citations": self._auto_citations(evidence), "confidence": self._conf(evidence), "data_used": []}
        except Exception as e:
            return {"answer": f"Error: {e}", "citations": [], "confidence": 0.0, "data_used": []}

    def _format(self, evidence):
        parts = []
        for i, ev in enumerate(evidence, 1):
            tool = ev.get("tool_used", "?")
            parts.append(f"--- Evidence #{i} ({tool}) ---")
            parts.append(f"Step: {ev.get('step_description', '')}")
            if ev.get("pages"):
                parts.append(f"Pages: {ev['pages']}")
            for r in ev.get("results", [])[:5]:
                if isinstance(r, dict):
                    if "text" in r:
                        parts.append(f"[Page {r.get('page', '?')}]: {r['text'][:500]}")
                    elif "result" in r:
                        parts.append(f"Calc: {r.get('function', '')}({r.get('parameters', {})}) = {r['result']}")
                    else:
                        parts.append(f"Row: {json.dumps(r, default=str)}")
            parts.append("")
        return "\n".join(parts)

    def _auto_citations(self, evidence):
        cites, seen = [], set()
        for ev in evidence:
            for r in ev.get("results", []):
                p = r.get("page")
                t = r.get("text", "")
                if p and p not in seen and len(t) >= 10:
                    cites.append({"page": p, "text": t[:200]})
                    seen.add(p)
        return cites[:5]

    def _conf(self, evidence):
        if not evidence:
            return 0.0
        return round(sum(e.get("confidence", 0.0) for e in evidence) / len(evidence), 2)