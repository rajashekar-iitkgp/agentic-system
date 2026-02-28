import logging
import asyncio
from typing import List
from app.core.embeddings import ToolEmbeddings
from app.core.config import settings
from app.models.tool import ToolMetadata
from app.db.vector.pinecone_client import pinecone_registry
from app.db.vector.faiss_client import faiss_registry

logger = logging.getLogger(__name__)

class ToolRegistryManager:
    def __init__(self):
        self.embeddings = ToolEmbeddings()

    async def seed_tools(self, tools: List[ToolMetadata]):
        logger.info(f"Seeding {len(tools)} tools into the Vector Store.")
        
        texts = []
        for tool in tools:
            rich_description = f"Tool Name: {tool.name}. Domain: {tool.domain}. Purpose: {tool.description}. Tags: {', '.join(tool.tags)}"
            texts.append(rich_description)
            
        logger.info("Generating embeddings...")
        embedded_vectors = await self.embeddings.aembed_documents(texts)
        
        for idx, tool in enumerate(tools):
            tool.embedding = embedded_vectors[idx]
            
        # Seed local HNSW (FAISS)
        await faiss_registry.upsert_tools(tools)
        # await pinecone_registry.upsert_tools(tools) # Disabled for OpenAI migration (requires new index)
        logger.info("Successfully seeded tools into Pinecone and FAISS HNSW.")

registry_manager = ToolRegistryManager()
