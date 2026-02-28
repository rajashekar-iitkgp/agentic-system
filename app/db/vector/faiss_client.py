import logging
import os
import pickle
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from app.core.config import settings
from app.models.tool import ToolMetadata

logger = logging.getLogger(__name__)

class FAISSToolRegistry:
    def __init__(self, index_path: str = "data/hnsw_index.faiss", metadata_path: str = "data/tool_metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        self.index = None
        self.tool_map: Dict[int, ToolMetadata] = {}  # Index ID to ToolMetadata
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.tool_map = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} tools.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        # IndexHNSWFlat is the standard FAISS HNSW implementation
        # M = 32 is a good balance between recall and memory
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        # efConstruction determines the graph quality at build time
        self.index.hnsw.efConstruction = 128
        # efSearch determines the search quality
        self.index.hnsw.efSearch = 64
        self.tool_map = {}
        logger.info("Created new empty FAISS HNSW index.")

    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.tool_map, f)
            logger.info("Saved FAISS index and metadata.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    async def upsert_tools(self, tools: List[ToolMetadata]):
        embeddings = []
        for tool in tools:
            if not tool.embedding:
                logger.warning(f"Tool {tool.name} has no embedding. Skipping.")
                continue
            embeddings.append(tool.embedding)
        
        if not embeddings:
            return

        vectors = np.array(embeddings).astype('float32')
        
        # In a real hierarchical system, we might add labels/clusters
        # For now, we add to HNSW directly. HNSW handles the hierarchy internally.
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        for i, tool in enumerate(tools):
            self.tool_map[start_id + i] = tool
            
        self.save_index()

    async def semantic_search(self, query_embedding: List[float], top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search the HNSW index
        distances, indices = self.index.search(query_vector, top_k * 2) # Get more for filtering
        
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            tool = self.tool_map.get(int(idx))
            if not tool: continue
            
            # Application-level filtering for Domain
            if domain_filter and tool.domain != domain_filter:
                continue
                
            matches.append({
                "id": tool.name,
                "score": float(dist), # FAISS L2 distance (smaller is better, usually)
                "metadata": {
                    "name": tool.name,
                    "description": tool.description,
                    "domain": tool.domain,
                    "tags": tool.tags,
                    "input_schema": str(tool.input_schema)
                }
            })
            
            if len(matches) >= top_k:
                break
                
        return matches

faiss_registry = FAISSToolRegistry()
