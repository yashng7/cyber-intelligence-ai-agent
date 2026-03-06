"""Citation lookup tool."""

from typing import Dict
import structlog

logger = structlog.get_logger()


def get_citation(page: int, keyword: str) -> Dict:
    from storage.vector_db import VectorStore
    store = VectorStore()

    try:
        results = store.query(query_text=keyword, n_results=10, where={"page": page})
        if not results:
            return {"page": page, "text": "", "found": False}

        best = results[0]
        return {"page": page, "text": best["text"], "found": True, "score": best["score"]}
    except Exception as e:
        logger.error("Citation failed", error=str(e))
        return {"page": page, "text": "", "found": False}