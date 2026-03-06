"""Planner Agent."""

import json
from typing import List, Optional
import structlog

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a planning agent. Break the user query into executable steps.

Tools: vector_search, table_lookup, math_calculator, citation_verifier
Rules: table_lookup for numbers/stats, vector_search for text, math_calculator for arithmetic, always end with citation_verifier. Max 6 steps.
Return ONLY a JSON array: [{"step_number":1,"action":"tool_name","description":"what","input":"query_or_params"}]"""


class PlannerAgent:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id
        from agents.gemini_client import create_model
        self.model = create_model(system_instruction=SYSTEM_PROMPT, temperature=0.1, max_output_tokens=1024)
        logger.info("PlannerAgent ready", trace_id=trace_id)

    def run(self, query: str) -> List[dict]:
        from agents.gemini_client import generate_with_retry

        try:
            raw = generate_with_retry(self.model, f"Create plan for:\n\n{query}", trace_id=self.trace_id)

            clean = raw.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

            plan = json.loads(clean)
            if not isinstance(plan, list):
                raise ValueError("Not a list")

            valid = {"vector_search", "table_lookup", "math_calculator", "citation_verifier"}
            for s in plan:
                if s.get("action") not in valid:
                    s["action"] = "vector_search"

            return plan[:6]

        except Exception as e:
            logger.error("Planning failed", error=str(e), trace_id=self.trace_id)
            return [
                {"step_number": 1, "action": "vector_search", "description": "Search text", "input": query},
                {"step_number": 2, "action": "table_lookup", "description": "Check tables", "input": query},
                {"step_number": 3, "action": "citation_verifier", "description": "Verify", "input": {"keyword": query.split()[0] if query else "report"}}
            ]