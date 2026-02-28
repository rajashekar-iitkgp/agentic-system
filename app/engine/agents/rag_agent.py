import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.engine.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
        self.system_prompt = SystemMessage(content="""
        You are the Documentation Assistant. 
        Your job is to answer the user's question using ONLY the retrieved context. 
        If the context does not contain the answer, say "I cannot find the answer in the documentation."
        """)
        
    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        logger.info("RAG Agent activated.")
        
        # Enhanced Mock Context for common documentation queries
        context_msg = SystemMessage(content="""
        [RELEVANT DOCUMENTATION]:
        - To handle a dispute or chargeback: Navigate to the Resolution Center in your PayPal dashboard. Select the specific transaction and follow the 'Resolve Dispute' prompts.
        - Integration: Use the PayPal SDKs or REST APIs to automate payment flows.
        - Security: Always enable Two-Factor Authentication (2FA) for merchant accounts.
        """)
        
        response = await self.llm.ainvoke([self.system_prompt, context_msg] + list(messages))
        return {"messages": [response]}

rag_node = RAGAgent()
