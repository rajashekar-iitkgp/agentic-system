from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ToolRanker:
    @staticmethod
    def rerank_tools(retrieved_tools: List[Dict[str, Any]], previous_tool_error: str = None) -> List[Dict[str, Any]]:
        if not previous_tool_error:
            return retrieved_tools
            
        logger.info(f"Re-ranking tools based on previous error: '{previous_tool_error}'")
        prioritized = []
        regular = []
        
        error_lower = previous_tool_error.lower()
        needs_search = "missing" in error_lower or "id" in error_lower or "not found" in error_lower
        
        for tool in retrieved_tools:
            if needs_search and "search" in tool["name"].lower():
                prioritized.append(tool)
            else:
                regular.append(tool)
                
        reranked = prioritized + regular
        logger.debug(f"Tool Priority post-ranking: {[t['name'] for t in reranked]}")
        return reranked
