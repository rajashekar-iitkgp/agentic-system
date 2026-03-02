import logging
import psycopg
import sys
import json
from urllib.parse import urlparse, urlunparse
from app.core.config import settings

logger = logging.getLogger(__name__)

# Ensure correct event loop on Windows for psycopg3
if sys.platform == "win32":
    import asyncio
    try:
        if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

async def save_response_to_db(session_id: str, user_query: str, agent_response: str, domain: str = None, tools: list = None):
    """
    Saves the final AI response to the PostgreSQL database in the background.
    """
    def _candidate_db_urls(db_url: str):
        urls = [db_url]
        try:
            parsed = urlparse(db_url)
            host = parsed.hostname
            port = parsed.port
            if host in {"127.0.0.1", "localhost"} and port:
                host_gateway = "host.docker.internal"
                netloc = parsed.netloc
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
        out = []
        for u in urls:
            if u not in out:
                out.append(u)
        return out

    db_url = settings.DATABASE_URL
    
    try:
        last_err = None
        for url in _candidate_db_urls(db_url):
            try:
                async with await psycopg.AsyncConnection.connect(url) as conn:
                    async with conn.cursor() as cur:
                        tools_str = json.dumps(tools) if tools else None
                        await cur.execute(
                            """
                            INSERT INTO agent_responses (session_id, user_query, agent_response, domain, retrieved_tools)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (session_id, user_query, agent_response, domain, tools_str),
                        )
                        await conn.commit()
                        logger.info(f"Successfully mirrored response for session {session_id} to DB.")
                        return
            except Exception as e:
                last_err = e
                continue
        raise last_err
    except Exception as e:
        logger.error(f"Failed to mirror response to DB: {e}")
