"""Load data into ChromaDB and DuckDB."""

import os
import sys
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import structlog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.vector_db import VectorStore
from storage.structured_store import StructuredStore

logger = structlog.get_logger()


def load_all_data(chunks: List[Dict], tables: List[Dict], embedding_model="all-MiniLM-L6-v2"):
    logger.info("Loading data", chunks=len(chunks), tables=len(tables))

    vector_store = VectorStore()
    structured_store = StructuredStore()

    # Embeddings
    logger.info("Loading embedding model")
    model = SentenceTransformer(embedding_model)

    logger.info("Generating embeddings")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Prepare for upsert
    prepared = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{chunk['document_source']}_page{chunk['page']}_chunk{chunk['chunk_index']}"
        prepared.append({
            "id": chunk_id,
            "text": chunk["text"],
            "embedding": embeddings[i].tolist(),
            "metadata": {
                "page": chunk["page"],
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
                "document_source": chunk["document_source"],
                "token_count": chunk["token_count"]
            }
        })

    # Upsert to ChromaDB
    vector_store.upsert(prepared)

    # Verify
    final_count = vector_store.collection.count()
    logger.info(f"VERIFY: ChromaDB has {final_count} chunks after upsert")

    if final_count > 0:
        test = vector_store.query("cybersecurity", n_results=1)
        logger.info(f"VERIFY: Test query returned {len(test)} results")
    else:
        logger.error("CRITICAL: ChromaDB has 0 chunks!")

    # Load tables to DuckDB
    for table in tables:
        try:
            structured_store.create_table(table["table_id"], table["dataframe"])
        except Exception as e:
            logger.error("Table load failed", table_id=table["table_id"], error=str(e))

    structured_store.close()
    logger.info("Data loading complete")