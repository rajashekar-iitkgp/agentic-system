import logging
import os
import pickle
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.models.tool import ToolMetadata

logger = logging.getLogger(__name__)

class FAISSToolRegistry:
    def __init__(self, index_path: str = "data/hnsw_index.faiss", metadata_path: str = "data/tool_metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 1536
        self.index = None
        self.tool_map: Dict[int, ToolMetadata] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_id_map: List[int] = []  # maps BM25 doc index → tool_map key

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.tool_map = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} tools.")
                self._rebuild_bm25()
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 128
        self.index.hnsw.efSearch = 64
        self.tool_map = {}
        self.bm25 = None
        self.bm25_id_map = []
        logger.info("Created new empty FAISS HNSW index.")

    def _rebuild_bm25(self):
        if not self.tool_map:
            self.bm25 = None
            self.bm25_id_map = []
            return
        
        self.bm25_id_map = sorted(self.tool_map.keys())
        corpus = []
        for idx in self.bm25_id_map:
            tool = self.tool_map[idx] 
            # Combine name + description tokens for keyword scoring
            text = f"{tool.name} {tool.description} {' '.join(tool.tags or [])}"
            tokens = text.lower().split()
            corpus.append(tokens)
        
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(corpus)} tools.")

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
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        valid_tools = [t for t in tools if t.embedding]
        for i, tool in enumerate(valid_tools):
            self.tool_map[start_id + i] = tool
        
        # Rebuild BM25 in sync with FAISS    
        self._rebuild_bm25()
        self.save_index()

    def _bm25_search(self, query: str, top_k: int, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.bm25_id_map:
            return []
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        # Pair scores with tool_map keys
        ranked = sorted(
            [(self.bm25_id_map[i], scores[i]) for i in range(len(scores))],
            key=lambda x: x[1], reverse=True
        )
        
        results = []
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        for tool_id, score in ranked:
            tool = self.tool_map.get(tool_id)
            if not tool:
                continue
            if domain_filter and tool.domain != domain_filter:
                continue
            # Normalize to 0-1
            results.append({"tool_id": tool_id, "score": score / max_score if max_score else 0})
            if len(results) >= top_k * 2:
                break
        
        return results

    async def semantic_search(self, query_embedding: List[float], top_k: int = 5, domain_filter: Optional[str] = None, action_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k * 3)
        
        # FAISS results with domain/type filtering
        faiss_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            tool = self.tool_map.get(int(idx))
            if not tool:
                continue
            if domain_filter and tool.domain != domain_filter:
                continue
            if action_type_filter and action_type_filter not in tool.tags:
                continue
            faiss_results.append({"tool_id": int(idx), "score": float(dist)})

        return [
            {
                "id": self.tool_map[r["tool_id"]].name,
                "score": r["score"],
                "metadata": {
                    "name": self.tool_map[r["tool_id"]].name,
                    "description": self.tool_map[r["tool_id"]].description,
                    "domain": self.tool_map[r["tool_id"]].domain,
                    "tags": self.tool_map[r["tool_id"]].tags,
                    "input_schema": str(self.tool_map[r["tool_id"]].input_schema)
                }
            }
            for r in faiss_results[:top_k]
            if r["tool_id"] in self.tool_map
        ]

    async def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        RRF_K = 60  # Standard RRF constant — higher = smoother fusion

        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k * 3)
        
        faiss_ranked = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            tool = self.tool_map.get(int(idx))
            if not tool:
                continue
            if domain_filter and tool.domain != domain_filter:
                continue
            faiss_ranked.append(int(idx))

        # BM25 Keyword Search
        bm25_results = self._bm25_search(query, top_k, domain_filter)
        bm25_ranked = [r["tool_id"] for r in bm25_results]

        # Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        for rank, tool_id in enumerate(faiss_ranked):
            rrf_scores[tool_id] = rrf_scores.get(tool_id, 0) + 1 / (RRF_K + rank + 1)
        for rank, tool_id in enumerate(bm25_ranked):
            rrf_scores[tool_id] = rrf_scores.get(tool_id, 0) + 1 / (RRF_K + rank + 1)

        # Sort by combined RRF score (higher is better)
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for tool_id in sorted_ids[:top_k]:
            tool = self.tool_map.get(tool_id)
            if not tool:
                continue
            results.append({
                "id": tool.name,
                "score": rrf_scores[tool_id],
                "metadata": {
                    "name": tool.name,
                    "description": tool.description,
                    "domain": tool.domain,
                    "tags": tool.tags,
                    "input_schema": str(tool.input_schema)
                }
            })
        
        logger.info(f"Hybrid search returned {len(results)} tools (BM25 + FAISS RRF).")
        return results

faiss_registry = FAISSToolRegistry()