from typing import List, Dict, Any

class ToolFilters:
    """
    Hard-filtering layer applied BEFORE semantic search.
    Improves retrieval precision by constraining the search space.
    """
    
    @staticmethod
    def get_domain_from_intent(user_intent: str) -> str:
        """
        Extracts a broad domain (e.g., 'payments', 'system', 'documentation') 
        from the user's intent to pass as a metadata filter to the Vector DB.
        """
        intent_lower = user_intent.lower()
        if any(kw in intent_lower for kw in ["invoice", "pay", "charge", "refund"]):
            return "payments"
        elif any(kw in intent_lower for kw in ["how to", "guide", "documentation", "docs"]):
            return "documentation"
        elif any(kw in intent_lower for kw in ["status", "system", "logs", "error rate"]):
            return "system"
        return "general"
