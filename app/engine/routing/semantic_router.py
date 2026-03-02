import logging
from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel, Field
from app.core.embeddings import ToolEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.db.vector.pg_client import pg_registry
# from app.db.vector.faiss_client import faiss_registry
from app.models.tool import ToolMetadata
from app.engine.routing.compressor import schema_compressor

logger = logging.getLogger(__name__)

class RankerSelection(BaseModel):
    selected_tool_names: List[str] = Field(
        ..., 
        description="The names of the tools that are MOST relevant to the user query, in order of relevance."
    )

class SemanticRouter:
    def __init__(self):
        self.embeddings = ToolEmbeddings()
        self.ranker_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            google_api_key=settings.GEMINI_API_KEY,
        )
        self.structured_ranker = self.ranker_llm.with_structured_output(RankerSelection)

    async def retrieve_tools_for_intent(self, user_query: str, domain_filter: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Two-Stage routing for query: '{user_query}'")

        # --- Stage 1: Hybrid retrieval (semantic + keyword) with quota-safe fallback ---
        quota_exhausted = False

        try:
            try:
                query_vector = await self.embeddings.aembed_query(user_query)
            except Exception as embed_err:
                msg = str(embed_err).lower()
                logger.error(f"Embedding generation failed: {embed_err}")
                if (
                    "insufficient_quota" in msg
                    or "you exceeded your current quota" in msg
                    or "429" in msg
                    or "quota" in msg
                    or "rate limit" in msg
                    or "resource_exhausted" in msg
                ):
                    # When LLM quota is exhausted, fall back to pure keyword search by
                    # passing an empty embedding into the SQL layer.
                    logger.warning("Embedding quota exhausted. Falling back to keyword-only hybrid search.")
                    quota_exhausted = True
                    query_vector = []
                else:
                    # Non-quota errors should still surface.
                    raise

            # If the embedding layer itself returned an empty vector (our quota-safe behavior),
            # treat this as an exhausted semantic signal and rely purely on SQL keyword search.
            if not query_vector:
                quota_exhausted = True

            candidates = await pg_registry.hybrid_search(
                query=user_query,
                query_embedding=query_vector,
                top_k=20,
                domain_filter=domain_filter,
            )
            logger.info(f"Stage 1: Retrieved {len(candidates)} candidates.")
        except Exception as e:
            logger.error(f"Stage 1 hybrid search failed: {e}")
            return []

        if not candidates:
            return []

        # --- Stage 2: Conditional re-ranking ---
        # If we know quota is exhausted for embeddings, skip the LLM ranker as well and
        # trust the hybrid SQL scores to avoid further failures.
        use_llm_ranker = not quota_exhausted
        if len(candidates) > 1:
            top_score = candidates[0].get("score", 0)
            second_score = candidates[1].get("score", 0)
            score_gap = top_score - second_score
            
            if score_gap > 0.005:
                logger.info(f"High Confidence (Gap: {score_gap:.4f}). Skipping LLM Re-ranking.")
                use_llm_ranker = False
            else:
                logger.info(f"Ambiguity detected (Gap: {score_gap:.4f}). Invoking LLM Re-ranker.")
        elif len(candidates) == 1:
            logger.info("Single candidate found. Skipping LLM Re-ranking.")
            use_llm_ranker = False

        if use_llm_ranker:
            selected_tools = await self._rerank_tools(user_query, candidates, k)
        else:
            selected_tools = [c['metadata'] for c in candidates[:k]]
        
        retrieved_tools = []
        for tool_meta in selected_tools:
            import json
            raw_schema = tool_meta.get("input_schema")
            if raw_schema is None or raw_schema in ("null", ""):
                schema_dict = {}
            elif isinstance(raw_schema, dict):
                schema_dict = raw_schema
            else:
                try:
                    schema_dict = json.loads(raw_schema) if isinstance(raw_schema, str) else {}
                except Exception:
                    schema_dict = {}
                
            compressed = schema_compressor.compress_tool_definition(
                name=tool_meta.get("name"),
                description=tool_meta.get("description", ""),
                schema=schema_dict
            )
            
            retrieved_tools.append({
                "name": compressed["n"], 
                "description": compressed["d"], 
                "input_schema": compressed["s"]
            })
            
        logger.info(f"Stage 2: Ranker selected {len(retrieved_tools)} tools.")
        return retrieved_tools

    async def _rerank_tools(self, query: str, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        candidate_info = "\n".join([f"- Name: {c['metadata']['name']} | Description: {c['metadata']['description']}" for c in candidates])
        prompt = f"""
        You are an expert Tool Selection Ranker.
        User Query: "{query}"

        Below is a list of candidate tools retrieved via keyword and vector search. 
        Analyze the query and select the {k} most relevant tools that can DIRECTLY help fulfill the user's request.
        
        Candidate Tools:
        {candidate_info}

        Return only the tool names in order of relevance.
        """
        try:
            logger.info("Executing Stage 2 Re-ranking...")
            selection: RankerSelection = await self.structured_ranker.ainvoke(prompt)
            ranked_names = selection.selected_tool_names[:k]
            final_selection = []
            name_to_meta = {c['metadata']['name']: c['metadata'] for c in candidates}
            
            for name in ranked_names:
                if name in name_to_meta:
                    final_selection.append(name_to_meta[name])
            return final_selection
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}. Falling back to Stage 1 top results.")
            return [c['metadata'] for c in candidates[:k]]

router = SemanticRouter()
