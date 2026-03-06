#!/usr/bin/env python3
"""ETL Pipeline entry point."""

import os
import sys
import time
import json
import structlog

structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.BoundLogger, context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def find_pdf(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    pdfs = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdfs:
        logger.error("No PDF found in ./data/")
        sys.exit(1)
    path = os.path.join(data_dir, pdfs[0])
    logger.info("Found PDF", path=path)
    return path


def main():
    pdf_path = find_pdf()
    start = time.time()
    summary = {"pdf_path": pdf_path, "text_blocks": 0, "tables_extracted": 0, "chunks_created": 0, "embeddings_loaded": 0, "tables_loaded": 0, "errors": []}

    # Step 1: Extract text
    logger.info("=" * 50)
    logger.info("STEP 1: Extract text")
    try:
        from etl.pdf_extractor import extract_pdf_content
        content = extract_pdf_content(pdf_path)
        summary["text_blocks"] = len(content)
        logger.info(f"Extracted {len(content)} blocks")
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        summary["errors"].append(str(e))
        content = []

    # Step 2: Extract tables
    logger.info("=" * 50)
    logger.info("STEP 2: Extract tables")
    try:
        from etl.table_extractor import extract_tables
        tables = extract_tables(pdf_path)
        summary["tables_extracted"] = len(tables)
    except Exception as e:
        logger.error(f"Table extraction failed: {e}")
        summary["errors"].append(str(e))
        tables = []

    # Step 3: Chunk
    logger.info("=" * 50)
    logger.info("STEP 3: Chunk text")
    try:
        from etl.chunking_pipeline import chunk_documents
        chunks = chunk_documents(content)
        summary["chunks_created"] = len(chunks)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        summary["errors"].append(str(e))
        chunks = []

    # Step 4: Load
    logger.info("=" * 50)
    logger.info("STEP 4: Load into stores")
    try:
        from etl.load_vector_store import load_all_data
        load_all_data(chunks, tables)
        summary["embeddings_loaded"] = len(chunks)
        summary["tables_loaded"] = len(tables)
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        summary["errors"].append(str(e))

    elapsed = time.time() - start
    summary["elapsed_seconds"] = round(elapsed, 1)

    logger.info("=" * 50)
    logger.info("ETL COMPLETE")
    logger.info(f"  Blocks:  {summary['text_blocks']}")
    logger.info(f"  Tables:  {summary['tables_extracted']}")
    logger.info(f"  Chunks:  {summary['chunks_created']}")
    logger.info(f"  Errors:  {len(summary['errors'])}")
    logger.info(f"  Time:    {summary['elapsed_seconds']}s")

    with open("./data/etl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if summary["chunks_created"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()