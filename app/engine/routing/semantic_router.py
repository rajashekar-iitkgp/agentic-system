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
    """
    Core innovation layer: Determines WHICH tools from the 500+ available 
    should be dynamically injected into the agent's context.
    """
    def __init__(self):
        load_dotenv()
        self.embeddings = GeminiEmbeddings(model_name="models/gemini-embedding-001")

    async def retrieve_tools_for_intent(
        self, 
        user_query: str, 
        domain_filter: Optional[str] = None, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        1. Embeds the user's intent.
        2. Queries Pinecone for the top K most semantically relevant tools.
        3. Returns the schema representations ready for injection.
        """
        logger.info(f"Semantically routing tools for query: '{user_query}' (Domain: {domain_filter})")
        
        # 1. Embed Intent (Async)
        query_vector = await self.embeddings.aembed_query(user_query)
        
        # 2. Vector Search
        matches = await pinecone_registry.semantic_search(
            query_embedding=query_vector,
            top_k=k,
            domain_filter=domain_filter
        )
        
        retrieved_tools = []
        for match in matches:
            meta = match["metadata"]
            
            # The input schema was stored as a stringified dict JSON in Pinecone
            import ast
            try:
                schema_dict = ast.literal_eval(meta.get("input_schema", "{}"))
            except:
                schema_dict = {}
                
            retrieved_tools.append({
                "name": meta.get("name"),
                "description": meta.get("description"),
                "input_schema": schema_dict
            })
            
        logger.info(f"Router selected {len(retrieved_tools)} tools.")
        return retrieved_tools

# Singleton instance
router = SemanticRouter()
