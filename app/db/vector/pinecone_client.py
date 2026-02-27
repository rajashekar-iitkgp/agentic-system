import logging
from typing import List, Dict, Any, Optional
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from app.core.config import settings
from app.models.tool import ToolMetadata

logger = logging.getLogger(__name__)

class PineconeToolRegistry:
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            logger.warning("PINECONE_API_KEY not found. Pinecone client will not initialize.")
            self.pc = None
            self.index = None
            return

        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME

        # Ensure index exists with correct dimension
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name in existing_indexes:
            idx_desc = self.pc.describe_index(self.index_name)
            if idx_desc.dimension != 3072:
                logger.warning(f"Index '{self.index_name}' has wrong dimension ({idx_desc.dimension}). Recreating...")
                self.pc.delete_index(self.index_name)
                existing_indexes.remove(self.index_name)

        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index '{self.index_name}' with 3072 dimensions...")
            self.pc.create_index(
                name=self.index_name,
                dimension=3072,  # models/text-embedding-004
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT or "us-east-1"
                )
            )
        
        self.index = self.pc.Index(self.index_name)

    async def upsert_tools(self, tools: List[ToolMetadata]):
        if not self.index:
            raise ValueError("Pinecone client not initialized.")

        vectors = []
        for tool in tools:
            if not tool.embedding:
                raise ValueError(f"Tool {tool.name} missing embedding before upsert.")
            
            # Metadata filterable fields
            metadata = {
                "name": tool.name,
                "description": tool.description,
                "domain": tool.domain,
                "tags": tool.tags,
                # JSON stringified schema for retrieval payload
                "input_schema": str(tool.input_schema) 
            }
            vectors.append({
                "id": tool.name,
                "values": tool.embedding,
                "metadata": metadata
            })
            
        # Batch upsert
        self.index.upsert(vectors=vectors)
        logger.info(f"Upserted {len(tools)} tools into Pinecone.")

    async def semantic_search(self, query_embedding: List[float], top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.index:
            raise ValueError("Pinecone client not initialized.")
            
        filter_dict = {}
        if domain_filter:
            filter_dict["domain"] = domain_filter
            
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Reconstruct into a list of matched tools
        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
            
        return matches

# Singleton instance
pinecone_registry = PineconeToolRegistry()
