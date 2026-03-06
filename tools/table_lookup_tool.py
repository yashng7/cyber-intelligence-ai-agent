"""Table lookup tool."""

from typing import List, Dict, Optional
import structlog

logger = structlog.get_logger()


def table_lookup(table_id: Optional[str] = None, filters: Optional[Dict] = None, sql: Optional[str] = None) -> List[Dict]:
    from storage.structured_store import StructuredStore
    store = StructuredStore()

    try:
        if sql:
            results = store.query_sql(sql)
            store.close()
            return results

        if not table_id:
            tables = store.list_tables()
            store.close()
            return [{"table_name": t} for t in tables]

        clean_id = table_id.replace("-", "_").replace(" ", "_")
        where = ""
        if filters:
            conditions = []
            for col, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                else:
                    conditions.append(f"{col} = {val}")
            where = " WHERE " + " AND ".join(conditions)

        results = store.query_sql(f"SELECT * FROM {clean_id}{where}")
        store.close()
        return results
    except Exception as e:
        logger.error("Table lookup failed", error=str(e))
        store.close()
        return []