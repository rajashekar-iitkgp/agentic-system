import re
import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class DomainRouter:
    
    DOMAINS = ["payments", "disputes", "reporting", "auth"]
    
    KEYWORD_MAP = {
        "payments": [r"invoice", r"refund", r"pay", r"bill", r"transaction"],
        "disputes": [r"dispute", r"evidence", r"claim", r"appeal"],
        "reporting": [r"sales", r"balance", r"summary", r"report", r"volume"],
        "auth": [r"token", r"login", r"authentication", r"access"]
    }

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=settings.GEMINI_API_KEY)

    async def classify(self, query: str) -> str:
        for domain, patterns in self.KEYWORD_MAP.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    logger.info(f"Domain Router: Keyword match for {query} -> {domain}")
                    return domain
                    
        logger.info(f"Domain Router: Falling back to LLM for {query}")
        prompt = f"""
        Classify the following user query into exactly ONE of these domains: {', '.join(self.DOMAINS)}.
        If you are unsure, default to 'payments'.
        
        Query: {query}
        
        Domain:"""
        
        response = await self.llm.ainvoke(prompt)
        domain = response.content.strip().lower()
        
        if domain not in self.DOMAINS:
            domain = "payments"
            
        logger.info(f"Domain Router: LLM classified {query} -> {domain}")
        return domain

domain_router = DomainRouter()
