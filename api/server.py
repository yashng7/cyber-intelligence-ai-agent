"""FastAPI server."""

import json
import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
    wrapper_class=structlog.BoundLogger, context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()
os.makedirs("logs", exist_ok=True)
executor = ThreadPoolExecutor(max_workers=2)


class QueryRequest(BaseModel):
    query: str


class CitationResponse(BaseModel):
    page: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    trace_id: str
    verified: bool
    confidence: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    collections: Optional[dict] = None


app = FastAPI(title="Document Intelligence System", version="1.0.0")


def save_trace(trace):
    try:
        with open("logs/agent_traces.json", "a") as f:
            f.write(json.dumps(trace, default=str) + "\n")
    except Exception as e:
        logger.error("Trace save failed", error=str(e))


def load_trace(trace_id):
    try:
        if not os.path.exists("logs/agent_traces.json"):
            return None
        with open("logs/agent_traces.json") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    if t.get("trace_id") == trace_id:
                        return t
    except Exception:
        pass
    return None


def run_pipeline(query, trace_id):
    trace = {
        "trace_id": trace_id, "query": query, "plan": [], "tools_used": [],
        "retrieved_documents": [], "calculations": [], "final_answer": "",
        "citations": [], "verified": False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    try:
        from agents.planner_agent import PlannerAgent
        from agents.retrieval_agent import RetrievalAgent
        from agents.reasoning_agent import ReasoningAgent
        from agents.verification_agent import VerificationAgent

        # Plan
        planner = PlannerAgent(trace_id=trace_id)
        plan = planner.run(query)
        trace["plan"] = plan

        # Retrieve
        retriever = RetrievalAgent(trace_id=trace_id)
        all_evidence = []
        for step in plan:
            action = step.get("action", "")
            trace["tools_used"].append(action)
            if action == "math_calculator":
                ev = retriever.execute_math(step, all_evidence)
                if ev.get("results"):
                    trace["calculations"].append(ev["results"])
            else:
                ev = retriever.run(step)
                for r in ev.get("results", []):
                    if isinstance(r, dict) and "text" in r:
                        trace["retrieved_documents"].append({"page": r.get("page"), "text": r.get("text", "")[:200], "tool": action})
            all_evidence.append(ev)

        # Reason
        reasoner = ReasoningAgent(trace_id=trace_id)
        result = reasoner.run(query, all_evidence)
        answer = result.get("answer", "No answer")
        citations = result.get("citations", [])
        confidence = result.get("confidence", 0.0)
        trace["final_answer"] = answer
        trace["citations"] = citations

        # Verify
        verifier = VerificationAgent(trace_id=trace_id)
        ver = verifier.run(answer, all_evidence, citations)
        trace["verified"] = ver.get("verified", False)
        trace["verification_details"] = ver
        if not trace["verified"]:
            confidence = min(confidence, 0.5)
        trace["confidence"] = confidence

        save_trace(trace)
        return trace

    except Exception as e:
        logger.error("Pipeline failed", error=str(e), trace_id=trace_id)
        trace["error"] = str(e)
        trace["final_answer"] = f"Error: {e}"
        save_trace(trace)
        return trace


@app.get("/health", response_model=HealthResponse)
async def health():
    info = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        from storage.vector_db import VectorStore
        from storage.structured_store import StructuredStore
        vs = VectorStore()
        ss = StructuredStore()
        info["collections"] = {"vector_store_count": vs.collection.count(), "structured_tables": ss.list_tables()}
        ss.close()
    except Exception as e:
        info["collections"] = {"error": str(e)}
    return info


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(400, "Empty query")

    trace_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    trace = await loop.run_in_executor(executor, run_pipeline, q, trace_id)

    if "error" in trace and trace.get("final_answer", "").startswith("Error:"):
        raise HTTPException(500, trace["error"])

    cites = []
    for c in trace.get("citations", []):
        if isinstance(c, dict) and "page" in c and "text" in c:
            cites.append(CitationResponse(page=c["page"], text=c["text"]))
    if not cites:
        for d in trace.get("retrieved_documents", []):
            if d.get("page") and d.get("text"):
                cites.append(CitationResponse(page=d["page"], text=d["text"]))
                if len(cites) >= 2:
                    break

    return QueryResponse(
        answer=trace.get("final_answer", "No answer"),
        citations=cites, trace_id=trace_id,
        verified=trace.get("verified", False),
        confidence=trace.get("confidence", 0.0)
    )


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    t = load_trace(trace_id)
    if not t:
        raise HTTPException(404, "Not found")
    return t


@app.get("/traces")
async def list_traces(limit: int = 20):
    traces = []
    try:
        if os.path.exists("logs/agent_traces.json"):
            with open("logs/agent_traces.json") as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line.strip())
                        traces.append({"trace_id": t.get("trace_id"), "query": t.get("query"), "verified": t.get("verified"), "timestamp": t.get("timestamp")})
    except Exception:
        pass
    traces.reverse()
    return traces[:limit]


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, timeout_keep_alive=300)