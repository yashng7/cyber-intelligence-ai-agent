"""ChromaDB persistent vector store."""

import os
import chromadb
from typing import List, Dict, Optional
import structlog

logger = structlog.get_logger()


class VectorStore:
    def __init__(self, persist_directory="./chroma_store", collection_name="cyber_ireland_chunks"):
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        logger.info("VectorStore ready", path=self.persist_directory, count=self.collection.count())

    def upsert(self, chunks: List[Dict]) -> None:
        if not chunks:
            return

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c["id"] for c in batch]
            embeddings = [c["embedding"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = []
            for c in batch:
                meta = {}
                for k, v in c["metadata"].items():
                    if v is None:
                        meta[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)

            self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

        logger.info("Upserted", count=len(chunks), total=self.collection.count())

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        count = self.collection.count()
        if count == 0:
            return []

        kwargs = {"query_texts": [query_text], "n_results": min(n_results, count)}
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.error("Query failed", error=str(e))
            return []

        formatted = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            formatted.append({
                "text": docs[i],
                "page": meta.get("page"),
                "section": meta.get("section"),
                "score": round(max(0, 1.0 - dist), 3),
                "metadata": meta
            })

        return formatted