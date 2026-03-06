"""Retrieval Agent."""

import json
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger()


class RetrievalAgent:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id
        from tools.vector_search_tool import vector_search
        from tools.table_lookup_tool import table_lookup
        self.tools = {"vector_search": vector_search, "table_lookup": table_lookup}

        try:
            from storage.structured_store import StructuredStore
            s = StructuredStore()
            self.available_tables = s.list_tables()
            s.close()
        except Exception:
            self.available_tables = []

        logger.info("RetrievalAgent ready", tables=self.available_tables, trace_id=trace_id)

    def run(self, step: Dict) -> Dict:
        action = step.get("action", "vector_search")
        desc = step.get("description", "")
        inp = step.get("input", "")

        if action == "table_lookup":
            return self._table_lookup(inp, desc)
        elif action == "citation_verifier":
            return self._citation_lookup(inp, desc)
        else:
            return self._vector_search(inp, desc)

    def _vector_search(self, query: Any, desc: str) -> Dict:
        q = query if isinstance(query, str) else str(query)
        try:
            results = self.tools["vector_search"](q, n_results=5)
            conf = max((r.get("score", 0.0) for r in results), default=0.0)
            pages = list(set(r.get("page") for r in results if r.get("page")))
            return {"tool_used": "vector_search", "query_executed": q, "results": results, "pages": pages, "confidence": round(conf, 2), "step_description": desc}
        except Exception as e:
            return {"tool_used": "vector_search", "query_executed": q, "results": [], "pages": [], "confidence": 0.0, "error": str(e), "step_description": desc}

    def _table_lookup(self, inp: Any, desc: str) -> Dict:
        try:
            from agents.gemini_client import create_model, generate_with_retry
            tables = ", ".join(self.available_tables) if self.available_tables else "none"
            m = create_model(system_instruction=f"SQL generator. Tables: [{tables}]. Return ONLY SQL, no markdown.", temperature=0.0, max_output_tokens=256)
            inp_str = json.dumps(inp, default=str) if not isinstance(inp, str) else inp
            sql = generate_with_retry(m, f"Step: {desc}\nInput: {inp_str}", trace_id=self.trace_id)
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
            results = self.tools["table_lookup"](sql=sql)
            return {"tool_used": "table_lookup", "query_executed": sql, "results": results, "pages": [], "confidence": 0.8 if results else 0.0, "step_description": desc}
        except Exception as e:
            logger.error("Table lookup failed", error=str(e))
            return self._vector_search(str(inp), desc)

    def _citation_lookup(self, inp: Any, desc: str) -> Dict:
        from tools.citation_tool import get_citation
        try:
            page, keyword = None, ""
            if isinstance(inp, dict):
                page = inp.get("page")
                keyword = inp.get("keyword", "")
            elif isinstance(inp, str):
                keyword = inp
            if page:
                result = get_citation(page=int(page), keyword=keyword)
            else:
                results = self.tools["vector_search"](keyword, n_results=3)
                if results:
                    b = results[0]
                    result = {"page": b["page"], "text": b["text"], "found": True, "score": b["score"]}
                else:
                    result = {"page": 0, "text": "", "found": False}
            return {"tool_used": "citation_verifier", "query_executed": f"page={page},kw={keyword}", "results": [result], "pages": [result.get("page")] if result.get("found") else [], "confidence": result.get("score", 0.5) if result.get("found") else 0.0, "step_description": desc}
        except Exception as e:
            return {"tool_used": "citation_verifier", "query_executed": str(inp), "results": [], "pages": [], "confidence": 0.0, "error": str(e), "step_description": desc}

    def execute_math(self, step: Dict, prev: List[Dict]) -> Dict:
        from tools.math_tool import compute_cagr, compute_difference, compute_percentage_change
        from agents.gemini_client import create_model, generate_with_retry
        desc = step.get("description", "")
        inp = step.get("input", {})
        try:
            ev_text = json.dumps(prev, default=str)
            m = create_model(system_instruction="Extract math params. Functions: compute_cagr(beginning_value,ending_value,years), compute_difference(a,b), compute_percentage_change(old_value,new_value). Return JSON only: {\"function\":\"name\",\"param\":value,...}", temperature=0.0, max_output_tokens=256)
            raw = generate_with_retry(m, f"Desc: {desc}\nInput: {json.dumps(inp, default=str)}\n\nEvidence:\n{ev_text[:3000]}", trace_id=self.trace_id)
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
            params = json.loads(raw)
            func_name = params.pop("function", "compute_difference")
            funcs = {"compute_cagr": compute_cagr, "compute_difference": compute_difference, "compute_percentage_change": compute_percentage_change}
            result = funcs[func_name](**params)
            return {"tool_used": "math_calculator", "query_executed": f"{func_name}({params})", "results": [{"function": func_name, "parameters": params, "result": result}], "pages": [], "confidence": 1.0, "step_description": desc}
        except Exception as e:
            logger.error("Math failed", error=str(e))
            return {"tool_used": "math_calculator", "query_executed": str(inp), "results": [], "pages": [], "confidence": 0.0, "error": str(e), "step_description": desc}