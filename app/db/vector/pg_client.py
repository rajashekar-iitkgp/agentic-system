import logging
import json
import psycopg
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.models.tool import ToolMetadata

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

class PGToolRegistry:
    def __init__(self):
        self.db_url = settings.DATABASE_URL
        self._pool: Optional[AsyncConnectionPool] = None

    def pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            logger.info(f"Initializing Async Pool on {self.db_url}")
            self._pool = AsyncConnectionPool(
                self.db_url,
                min_size=1,
                max_size=10,
                open=False
            )
        return self._pool

    def _initialize_db_sync(self):
        try:
            with psycopg.connect(self.db_url, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS tool_registry (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT NOT NULL,
                        domain TEXT,
                        tags JSONB,
                        embedding vector(1536),
                        input_schema JSONB,
                        search_vector tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', name || ' ' || description)
                        ) STORED
                    );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS tool_search_idx ON tool_registry USING GIN(search_vector);")
                    cur.execute("CREATE INDEX IF NOT EXISTS tool_embedding_idx ON tool_registry USING hnsw (embedding vector_cosine_ops);")
            logger.info("PGToolRegistry schema verified (Sync).")
        except Exception as e:
            logger.error(f"Sync DB Init failed: {e}")

    async def upsert_tools(self, tools: List[ToolMetadata]):
        if not tools:
            return
            
        self._initialize_db_sync()
        
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    for tool in tools:
                        cur.execute("""
                        INSERT INTO tool_registry (name, description, domain, tags, embedding, input_schema)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                            description = EXCLUDED.description,
                            domain = EXCLUDED.domain,
                            tags = EXCLUDED.tags,
                            embedding = COALESCE(NULLIF(EXCLUDED.embedding, NULL), tool_registry.embedding),
                            input_schema = EXCLUDED.input_schema;
                        """, (
                            tool.name,
                            tool.description,
                            tool.domain,
                            json.dumps(tool.tags) if tool.tags else None,
                            tool.embedding,
                            json.dumps(tool.input_schema) if tool.input_schema else None
                        ))
                    conn.commit()
            logger.info(f"Successfully upserted {len(tools)} tools via Sync Connection.")
        except Exception as e:
            logger.error(f"Failed to upsert tools (Sync): {e}")

    async def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        p = self.pool()
        if not hasattr(self, "_pool_ready"):
            try:
                await p.open()
                self._pool_ready = True
            except Exception as e:
                # On Windows, if we hit Proactor issues, we log and try to continue 
                logger.warning(f"Async Pool Open warning: {e}. Check Windows Event Loop Policy if this persists.")
                self._pool_ready = True

        try:
            async with p.connection() as conn:
                async with conn.cursor() as cur:
                    sql = """
                    WITH semantic_search AS (
                        SELECT id, 1 - (embedding <=> %s::vector) AS vector_similarity
                        FROM tool_registry
                        WHERE (%s IS NULL OR domain = %s)
                        AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT 50
                    ),
                    keyword_search AS (
                        SELECT id, ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS keyword_score
                        FROM tool_registry
                        WHERE (
                            search_vector @@ plainto_tsquery('english', %s)
                            OR search_vector @@ to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|'))
                            OR name ILIKE '%%' || split_part(%s, ' ', 1) || '%%'
                        )
                        AND (%s IS NULL OR domain = %s)
                        LIMIT 50
                    )
                    SELECT 
                        t.id, t.name, t.description, t.domain, t.tags::text, t.input_schema::text,
                        COALESCE(s.vector_similarity, 0) + COALESCE(k.keyword_score, 0) AS combined_score
                    FROM tool_registry t
                    LEFT JOIN semantic_search s ON t.id = s.id
                    LEFT JOIN keyword_search k ON t.id = k.id
                    WHERE s.id IS NOT NULL OR k.id IS NOT NULL
                    ORDER BY combined_score DESC
                    LIMIT %s;
                    """
                    
                    safe_embedding = query_embedding if query_embedding else [0.0] * 1536
                    await cur.execute(sql, (
                        safe_embedding, domain_filter, domain_filter, safe_embedding,  # Semantic (4)
                        query, query, query, query, domain_filter, domain_filter,     # Keyword (6)
                        top_k                                                         # Limit (1)
                    ))
                    
                    rows = await cur.fetchall()
                    results = []
                    for row in rows:
                        results.append({
                            "id": row[0],
                            "score": float(row[6]),
                            "metadata": {
                                "name": row[1],
                                "description": row[2],
                                "domain": row[3],
                                "tags": json.loads(row[4]) if row[4] else [],
                                "input_schema": row[5]
                            }
                        })
                    return results
        except Exception as e:
            logger.error(f"Hybrid Search failed: {e}")
            return []

    async def close(self):
        if self._pool:
            await self._pool.close()
            logger.info("Postgres Connection Pool closed.")

pg_registry = PGToolRegistry()
