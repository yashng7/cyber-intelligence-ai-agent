"""Text chunking with metadata."""

from typing import List, Dict
import structlog

logger = structlog.get_logger()


def chunk_documents(content_blocks: List[Dict], chunk_size=500, overlap=100, document_source="cyber_ireland_report") -> List[Dict]:
    logger.info("Chunking", blocks=len(content_blocks))

    chunks = []
    chunk_index = 0
    text_blocks = [b for b in content_blocks if b["type"] == "text"]

    for block in text_blocks:
        text = block["content"]
        page = block["page"]
        section = block["section"]
        words = text.split()

        words_per_chunk = int(chunk_size / 1.3)
        words_overlap = int(overlap / 1.3)

        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            token_count = int(len(chunk_words) * 1.3)
            if token_count < 50:
                break

            chunks.append({
                "text": chunk_text,
                "page": page,
                "section": section,
                "chunk_index": chunk_index,
                "document_source": document_source,
                "token_count": token_count
            })
            chunk_index += 1

            start += (words_per_chunk - words_overlap)
            if end >= len(words):
                break

    logger.info("Chunking done", total_chunks=len(chunks))
    return chunks