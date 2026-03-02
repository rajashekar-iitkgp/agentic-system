import logging
import asyncio
from typing import List
from app.core.embeddings import ToolEmbeddings
from app.core.config import settings
from app.models.tool import ToolMetadata
from app.db.vector.pg_client import pg_registry
# from app.db.vector.faiss_client import faiss_registry
from app.tools.api_tools import ALL_TOOLS

logger = logging.getLogger(__name__)

class ToolRegistryManager:
    def __init__(self):
        self.embeddings = ToolEmbeddings()
        self.all_tool_implementations = ALL_TOOLS

    async def seed_tools(self, tools: List[ToolMetadata]):
        logger.info(f"Seeding {len(tools)} tools into the Unified SQL Store (PGVector).")
        
        texts = []
        for tool in tools:
            if not tool.tags or not any(t in ["read", "write"] for t in tool.tags):
                action_type = "read" if any(kw in tool.name.lower() for kw in ["get", "fetch", "check", "list"]) else "write"
                tool.tags.append(action_type)
                
            rich_description = f"Tool Name: {tool.name}. Domain: {tool.domain}. Purpose: {tool.description}. Tags: {', '.join(tool.tags)}"
            texts.append(rich_description)
            
        logger.info("Generating embeddings...")
        embedded_vectors = await self.embeddings.aembed_documents(texts)
        
        for idx, tool in enumerate(tools):
            tool.embedding = embedded_vectors[idx]
            
        await pg_registry.upsert_tools(tools)
        logger.info(f"Successfully seeded {len(tools)} tools into PostgreSQL.")

registry_manager = ToolRegistryManager()
