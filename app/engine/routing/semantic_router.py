import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from app.core.embeddings import ToolEmbeddings

from app.core.config import settings
from app.db.vector.faiss_client import faiss_registry
from app.models.tool import ToolMetadata

logger = logging.getLogger(__name__)

from app.engine.routing.compressor import schema_compressor

class SemanticRouter:
    def __init__(self):
        self.embeddings = ToolEmbeddings()

    async def retrieve_tools_for_intent(self, user_query: str, domain_filter: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Semantically routing tools (HNSW) for query: '{user_query}' (Domain: {domain_filter})")
        try:
            logger.info("Generating query embedding...")
            query_vector = await self.embeddings.aembed_query(user_query)
            logger.info(f"Query embedding generated. Size: {len(query_vector)}")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

        try:
            logger.info("Performing FAISS HNSW search...")
            matches = await faiss_registry.semantic_search(query_embedding=query_vector, top_k=k, domain_filter=domain_filter)
            logger.info(f"FAISS search returned {len(matches)} matches.")
        except Exception as e:
            logger.error(f"Failed FAISS search: {e}")
            raise
        
        retrieved_tools = []
        for match in matches:
            meta = match["metadata"]
            import ast
            try:
                schema_dict = ast.literal_eval(meta.get("input_schema", "{}"))
            except:
                schema_dict = {}
                
            # TOKEN OPTIMIZATION: Compress before adding to state
            compressed = schema_compressor.compress_tool_definition(
                name=meta.get("name"),
                description=meta.get("description", ""),
                schema=schema_dict
            )
            
            # Use original keys but compressed values for LLM stability
            retrieved_tools.append({
                "name": compressed["n"], 
                "description": compressed["d"], 
                "input_schema": compressed["s"]
            })
            
        logger.info(f"Router selected {len(retrieved_tools)} tools.")
        return retrieved_tools


router = SemanticRouter()
