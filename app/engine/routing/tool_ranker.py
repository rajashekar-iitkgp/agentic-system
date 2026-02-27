from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ToolRanker:
    """
    Re-ranking layer applied AFTER semantic search but BEFORE injection.
    Used to sort tools based on the current conversational state and recent errors.
    """
    
    @staticmethod
    def rerank_tools(
        retrieved_tools: List[Dict[str, Any]], 
        previous_tool_error: str = None
    ) -> List[Dict[str, Any]]:
        """
        Re-orders tools to prioritize fixes if an error occurred in the previous turn.
        For example, if `create_invoice` failed due to a missing customer ID, 
        we want to aggressively bump `search_customers` to the top of the context.
        """
        if not previous_tool_error:
            return retrieved_tools
            
        logger.info(f"Re-ranking tools based on previous error: '{previous_tool_error}'")
        
        # Simple heuristic: If the error mentions missing IDs or users, prioritize search tools.
        prioritized = []
        regular = []
        
        error_lower = previous_tool_error.lower()
        needs_search = "missing" in error_lower or "id" in error_lower or "not found" in error_lower
        
        for tool in retrieved_tools:
            if needs_search and "search" in tool["name"].lower():
                prioritized.append(tool)
            else:
                regular.append(tool)
                
        # Return prioritized search tools first, then the original sequence
        reranked = prioritized + regular
        logger.debug(f"Tool Priority post-ranking: {[t['name'] for t in reranked]}")
        return reranked
