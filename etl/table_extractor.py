"""Table extraction from PDF."""

import camelot
import tabula
import pandas as pd
from typing import List, Dict
import structlog

logger = structlog.get_logger()


def _identify_table_type(df: pd.DataFrame, page: int) -> str:
    if df.empty:
        return f"table_page_{page}"

    combined = " ".join(str(c).lower() for c in df.columns)
    if len(df) > 0:
        combined += " " + " ".join(str(v).lower() for v in df.iloc[0].values if pd.notna(v))

    if any(kw in combined for kw in ["region", "geographical", "location"]):
        return "regional_distribution"
    elif any(kw in combined for kw in ["employment", "jobs", "employees"]):
        return "employment_stats"
    elif any(kw in combined for kw in ["growth", "target", "2030", "forecast"]):
        return "growth_targets"
    elif any(kw in combined for kw in ["pure play", "diversified"]):
        return "firm_classification"
    else:
        return f"table_page_{page}"


def extract_tables(pdf_path: str) -> List[Dict]:
    logger.info("Extracting tables", path=pdf_path)
    results = []

    # Try Camelot
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        logger.info("Camelot found tables", count=len(tables))
        for idx, table in enumerate(tables):
            df = table.df
            df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            table_id = _identify_table_type(df, table.page)
            results.append({
                "page": table.page,
                "table_id": f"{table_id}_{idx}",
                "columns": df.columns.tolist(),
                "rows": df.values.tolist(),
                "dataframe": df
            })
    except Exception as e:
        logger.warning("Camelot failed", error=str(e))

    # Fallback to tabula if nothing found
    if not results:
        try:
            tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            logger.info("Tabula found tables", count=len(tables))
            for idx, df in enumerate(tables):
                if df.empty:
                    continue
                page = idx + 1
                table_id = _identify_table_type(df, page)
                results.append({
                    "page": page,
                    "table_id": f"{table_id}_{idx}",
                    "columns": df.columns.tolist(),
                    "rows": df.values.tolist(),
                    "dataframe": df
                })
        except Exception as e:
            logger.error("Tabula failed", error=str(e))

    logger.info("Table extraction done", tables_found=len(results))
    for t in results:
        logger.info(f"  Table: {t['table_id']} | Page {t['page']} | Rows: {len(t['rows'])}")

    return results