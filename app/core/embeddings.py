import os
import logging
import hashlib
import json
import redis
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class ToolEmbeddings(OpenAIEmbeddings):
    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__(
            model=model,
            api_key=settings.OPENAI_API_KEY
        )
        try:
            r = redis.from_url(settings.REDIS_URL, decode_responses=True)
            r.ping()
            object.__setattr__(self, 'redis_client', r)
            logger.info("Embedding Cache connected to Redis.")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for Embedding Cache: {e}. Caching disabled.")
            object.__setattr__(self, 'redis_client', None)

    def _get_query_hash(self, text: str) -> str:
        clean_text = text.strip().lower()
        return f"emb:{hashlib.sha256(clean_text.encode('utf-8')).hexdigest()}"

    async def aembed_query(self, text: str) -> List[float]:
        r_client = getattr(self, 'redis_client', None)
        if not r_client:
            return await super().aembed_query(text)

        cache_key = self._get_query_hash(text)
        try:
            cached_val = r_client.get(cache_key)
            if cached_val:
                logger.debug(f"Embedding Cache HIT: '{text[:20]}...'")
                return json.loads(cached_val)
        except Exception as e:
            logger.error(f"Redis cache error while reading: {e}")

        # Cache MISS: call OpenAI
        try:
            embedding = await super().aembed_query(text)
        except Exception as e:
            if "insufficient_quota" in str(e).lower():
                logger.error("--- OPENAI QUOTA EXCEEDED ---")
                logger.error("Please refill your OpenAI credits. System will continue but new embeddings won't be cached.")
                return [] # Return empty to handle gracefully
            raise
        
        # Save to Redis with 7-day TTL (604800 seconds) 
        try:
            r_client.set(cache_key, json.dumps(embedding), ex=604800)
            logger.debug(f"Embedding Cache MISS (Saved): '{text[:20]}...'")
        except Exception as e:
            logger.error(f"Redis cache error while writing: {e}")

        return embedding

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        r_client = getattr(self, 'redis_client', None)
        if not r_client:
            return await super().aembed_documents(texts)

        results = [None] * len(texts)
        missing_indices = []
        
        # Check cache for each document
        for idx, text in enumerate(texts):
            cache_key = self._get_query_hash(text)
            try:
                cached_val = r_client.get(cache_key)
                if cached_val:
                    results[idx] = json.loads(cached_val)
                else:
                    missing_indices.append(idx)
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
                missing_indices.append(idx)

        # Batch call OpenAI for missing embeddings
        if missing_indices:
            logger.info(f"Embedding Cache: {len(missing_indices)}/{len(texts)} MISSES.")
            missing_texts = [texts[i] for i in missing_indices]
            try:
                new_embeddings = await super().aembed_documents(missing_texts)
                
                # Save results and update cache
                for i, (idx, embedding) in enumerate(zip(missing_indices, new_embeddings)):
                    results[idx] = embedding
                    try:
                        cache_key = self._get_query_hash(texts[idx])
                        r_client.set(cache_key, json.dumps(embedding), ex=604800)
                    except Exception as e:
                        logger.error(f"Redis cache error during batch write: {e}")
            except Exception as e:
                if "insufficient_quota" in str(e).lower():
                    logger.error("--- OPENAI QUOTA EXCEEDED (BATCH) ---")
                    # Fill missing with zeros if blocked
                    for idx in missing_indices:
                        results[idx] = [0.0] * self.index.ntotal if hasattr(self, 'index') else [0.0] * 1536
                else:
                    raise

        return results
