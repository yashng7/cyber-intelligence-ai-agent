"""Vector search tool."""

from typing import List, Dict
import structlog

logger = structlog.get_logger()


def vector_search(query: str, n_results: int = 5) -> List[Dict]:
    from storage.vector_db import VectorStore
    store = VectorStore()
    results = store.query(query, n_results=n_results)
    logger.info("Vector search", query=query[:80], results=len(results))
    return results