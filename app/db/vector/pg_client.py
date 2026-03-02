import logging
import json
import asyncio
import psycopg
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.models.tool import ToolMetadata

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

def _candidate_db_urls(db_url: str) -> List[str]:
    """
    Keep the configured DATABASE_URL intact, but provide connectivity fallbacks for Docker.
    In Docker, 127.0.0.1 points to the container, not the host.
    """
    urls = [db_url]
    try:
        parsed = urlparse(db_url)
        host = parsed.hostname
        port = parsed.port
        if host in {"127.0.0.1", "localhost"} and port:
            # Windows/Mac Docker Desktop host gateway DNS.
            host_gateway = "host.docker.internal"
            netloc = parsed.netloc
            # Replace only the host part while preserving user:pass and port.
            # netloc forms: user:pass@host:port
            if "@" in netloc:
                creds, hostport = netloc.split("@", 1)
                if ":" in hostport:
                    _, p = hostport.rsplit(":", 1)
                    netloc_gateway = f"{creds}@{host_gateway}:{p}"
                else:
                    netloc_gateway = f"{creds}@{host_gateway}"
            else:
                netloc_gateway = f"{host_gateway}:{port}"

            urls.append(urlunparse(parsed._replace(netloc=netloc_gateway)))
    except Exception:
        pass
    # De-dupe while preserving order
    out: List[str] = []
    for u in urls:
        if u not in out:
            out.append(u)
    return out


def _hybrid_search_sync(
    db_url: str,
    query: str,
    query_embedding: List[float],
    top_k: int,
    domain_filter: Optional[str],
) -> List[Dict[str, Any]]:
    """Sync hybrid search to avoid Windows ProactorEventLoop + psycopg async issues."""
    safe_embedding = query_embedding if query_embedding else [0.0] * 1536
    sql = """
    WITH semantic_search AS (
        SELECT id, 1 - (embedding <=> %s::vector) AS vector_similarity
        FROM tool_registry
        WHERE (%s::text IS NULL OR domain = %s)
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
        AND (%s::text IS NULL OR domain = %s)
        LIMIT 50
    )
    SELECT t.id, t.name, t.description, t.domain, t.tags::text, t.input_schema::text,
        COALESCE(s.vector_similarity, 0) + COALESCE(k.keyword_score, 0) AS combined_score
    FROM tool_registry t
    LEFT JOIN semantic_search s ON t.id = s.id
    LEFT JOIN keyword_search k ON t.id = k.id
    WHERE s.id IS NOT NULL OR k.id IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT %s;
    """
    last_err: Optional[Exception] = None
    for url in _candidate_db_urls(db_url):
        try:
            with psycopg.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            safe_embedding,
                            domain_filter,
                            domain_filter,
                            safe_embedding,
                            query,
                            query,
                            query,
                            query,
                            domain_filter,
                            domain_filter,
                            top_k,
                        ),
                    )
                    rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "score": float(row[6]),
                    "metadata": {
                        "name": row[1],
                        "description": row[2],
                        "domain": row[3],
                        "tags": json.loads(row[4]) if row[4] else [],
                        "input_schema": row[5],
                    },
                }
                for row in rows
            ]
        except Exception as e:
            last_err = e
            continue

    logger.error(f"Hybrid Search (sync) failed: {last_err}")
    return []


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
        last_err: Optional[Exception] = None
        for url in _candidate_db_urls(self.db_url):
            try:
                with psycopg.connect(url, autocommit=True) as conn:
                    with conn.cursor() as cur:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        cur.execute(
                            """
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
                            """
                        )
                        cur.execute("CREATE INDEX IF NOT EXISTS tool_search_idx ON tool_registry USING GIN(search_vector);")
                        cur.execute("CREATE INDEX IF NOT EXISTS tool_embedding_idx ON tool_registry USING hnsw (embedding vector_cosine_ops);")
                logger.info("PGToolRegistry schema verified (Sync).")
                return
            except Exception as e:
                last_err = e
                continue
        logger.error(f"Sync DB Init failed: {last_err}")

    async def upsert_tools(self, tools: List[ToolMetadata]):
        if not tools:
            return
            
        self._initialize_db_sync()
        
        last_err: Optional[Exception] = None
        for url in _candidate_db_urls(self.db_url):
            try:
                with psycopg.connect(url) as conn:
                    with conn.cursor() as cur:
                        for tool in tools:
                            cur.execute(
                                """
                                INSERT INTO tool_registry (name, description, domain, tags, embedding, input_schema)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (name) DO UPDATE SET
                                    description = EXCLUDED.description,
                                    domain = EXCLUDED.domain,
                                    tags = EXCLUDED.tags,
                                    embedding = COALESCE(NULLIF(EXCLUDED.embedding, NULL), tool_registry.embedding),
                                    input_schema = EXCLUDED.input_schema;
                                """,
                                (
                                    tool.name,
                                    tool.description,
                                    tool.domain,
                                    json.dumps(tool.tags) if tool.tags else None,
                                    tool.embedding,
                                    json.dumps(tool.input_schema) if tool.input_schema else None,
                                ),
                            )
                        conn.commit()
                logger.info(f"Successfully upserted {len(tools)} tools via Sync Connection.")
                return
            except Exception as e:
                last_err = e
                continue
        logger.error(f"Failed to upsert tools (Sync): {last_err}")

    async def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Runs hybrid search in a thread to avoid Windows ProactorEventLoop + psycopg async issues."""
        self._initialize_db_sync()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                _hybrid_search_sync,
                self.db_url,
                query,
                query_embedding,
                top_k,
                domain_filter,
            )
        except Exception as e:
            logger.error(f"Hybrid Search failed: {e}")
            return []

    async def close(self):
        if self._pool:
            await self._pool.close()
            logger.info("Postgres Connection Pool closed.")

pg_registry = PGToolRegistry()
