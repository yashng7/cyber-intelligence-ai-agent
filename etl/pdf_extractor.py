"""PDF text extraction."""

import fitz
import pdfplumber
from typing import List, Dict
import structlog

logger = structlog.get_logger()


def extract_pdf_content(pdf_path: str) -> List[Dict]:
    logger.info("Extracting PDF", path=pdf_path)

    doc = fitz.open(pdf_path)
    results = []
    current_section = "Introduction"

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_texts = []

        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    if line_text.isupper() and len(line_text.split()) >= 3:
                        current_section = line_text
                        results.append({"page": page_num + 1, "type": "heading", "content": line_text, "section": current_section})
                    else:
                        page_texts.append(line_text)

        if page_texts:
            combined = " ".join(page_texts)
            results.append({"page": page_num + 1, "type": "text", "content": combined, "section": current_section})

            # Fallback for thin pages
            if len(combined.strip()) < 50:
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        if page_num < len(pdf.pages):
                            fallback = pdf.pages[page_num].extract_text() or ""
                            if len(fallback) > len(combined):
                                results[-1]["content"] = fallback
                except Exception:
                    pass

    doc.close()
    logger.info("Extraction done", blocks=len(results))
    return results