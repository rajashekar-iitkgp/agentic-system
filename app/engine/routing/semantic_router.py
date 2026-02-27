import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from app.core.embeddings import GeminiEmbeddings

from app.core.config import settings
from app.db.vector.pinecone_client import pinecone_registry
from app.models.tool import ToolMetadata

logger = logging.getLogger(__name__)

class SemanticRouter:
    def __init__(self):
        self.embeddings = GeminiEmbeddings(model_name="models/gemini-embedding-001")

    async def retrieve_tools_for_intent(self, user_query: str, domain_filter: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Semantically routing tools for query: '{user_query}' (Domain: {domain_filter})")
        query_vector = await self.embeddings.aembed_query(user_query)
        matches = await pinecone_registry.semantic_search(query_embedding=query_vector, top_k=k, domain_filter=domain_filter)
        
        retrieved_tools = []
        for match in matches:
            meta = match["metadata"]
            import ast
            try:
                schema_dict = ast.literal_eval(meta.get("input_schema", "{}"))
            except:
                schema_dict = {}
                
            retrieved_tools.append({"name": meta.get("name"), "description": meta.get("description"), "input_schema": schema_dict})
            
        logger.info(f"Router selected {len(retrieved_tools)} tools.")
        return retrieved_tools

router = SemanticRouter()
