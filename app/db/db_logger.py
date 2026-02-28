import logging
import psycopg
import sys
import json
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
    db_url = settings.DATABASE_URL
    
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                tools_str = json.dumps(tools) if tools else None
                
                await cur.execute(
                    """
                    INSERT INTO agent_responses (session_id, user_query, agent_response, domain, retrieved_tools)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, user_query, agent_response, domain, tools_str)
                )
                await conn.commit()
                logger.info(f"Successfully mirrored response for session {session_id} to DB.")
    except Exception as e:
        logger.error(f"Failed to mirror response to DB: {e}")
