"""Verification Agent."""

import json
import re
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()

SYSTEM_PROMPT = """Verify answer uses ONLY facts from evidence. Check each claim, page number, and number.
Return ONLY JSON: {"verified":bool,"issues":["problems"],"claim_checks":[{"claim":"x","supported":bool,"source":"y"}],"confidence":0.0-1.0}"""


class VerificationAgent:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id
        from agents.gemini_client import create_model
        self.model = create_model(system_instruction=SYSTEM_PROMPT, temperature=0.0, max_output_tokens=1024)

    def run(self, answer: str, evidence: List[Dict], citations: List[Dict]) -> Dict:
        from agents.gemini_client import generate_with_retry
        ev_text = self._format(evidence)
        try:
            raw = generate_with_retry(self.model, f"ANSWER:\n{answer}\n\nCITATIONS:\n{json.dumps(citations, default=str)}\n\nEVIDENCE:\n{ev_text}\n\nVerify.", trace_id=self.trace_id)
            clean = raw.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
            result = json.loads(clean)
            result.setdefault("verified", False)
            result.setdefault("issues", [])
            result.setdefault("claim_checks", [])
            result.setdefault("confidence", 0.5)
            return result
        except json.JSONDecodeError:
            return self._fallback(answer, evidence)
        except Exception as e:
            return {"verified": False, "issues": [str(e)], "claim_checks": [], "confidence": 0.0}

    def _format(self, evidence):
        parts = []
        for i, ev in enumerate(evidence, 1):
            parts.append(f"Evidence #{i} ({ev.get('tool_used', '?')}, pages {ev.get('pages', [])}):")
            for r in ev.get("results", []):
                if isinstance(r, dict):
                    if "text" in r:
                        parts.append(f"  {r['text'][:500]}")
                    if "result" in r:
                        parts.append(f"  {r.get('function', '')} = {r['result']}")
            parts.append("")
        return "\n".join(parts)

    def _fallback(self, answer, evidence):
        ev_text = ""
        for ev in evidence:
            for r in ev.get("results", []):
                if isinstance(r, dict):
                    ev_text += " " + r.get("text", "") + " " + str(r.get("result", ""))
        a_nums = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', answer.lower()))
        e_nums = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', ev_text.lower()))
        bad = a_nums - e_nums
        issues = [f"Unsupported numbers: {bad}"] if bad else []
        return {"verified": not issues, "issues": issues, "claim_checks": [], "confidence": 1.0 if not issues else 0.5}