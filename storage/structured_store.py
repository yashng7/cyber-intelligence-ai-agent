"""DuckDB structured data store."""

import os
import duckdb
import pandas as pd
from typing import List, Dict
import structlog

logger = structlog.get_logger()


class StructuredStore:
    def __init__(self, db_path="./data/structured.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        logger.info("StructuredStore ready", path=db_path)

    def create_table(self, table_id: str, df: pd.DataFrame) -> None:
        clean_id = table_id.replace("-", "_").replace(" ", "_")
        self.conn.execute(f"DROP TABLE IF EXISTS {clean_id}")
        self.conn.execute(f"CREATE TABLE {clean_id} AS SELECT * FROM df")
        count = self.conn.execute(f"SELECT COUNT(*) FROM {clean_id}").fetchone()[0]
        logger.info("Table created", table_id=clean_id, rows=count)

    def query_sql(self, sql: str) -> List[Dict]:
        try:
            result = self.conn.execute(sql).fetchdf()
            return result.to_dict("records")
        except Exception as e:
            logger.error("SQL failed", sql=sql[:100], error=str(e))
            raise

    def list_tables(self) -> List[str]:
        result = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchdf()
        return result["table_name"].tolist()

    def get_table_schema(self, table_name: str) -> List[Dict]:
        try:
            result = self.conn.execute(f"PRAGMA table_info({table_name})").fetchdf()
            return result.to_dict("records")
        except Exception:
            return []

    def close(self):
        self.conn.close()