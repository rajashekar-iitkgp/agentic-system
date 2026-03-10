import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.engine.state import AgentState
from app.core.config import settings
from app.core.embeddings import ToolEmbeddings
from app.db.vector.pg_client import pg_registry

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.GENAI_MODEL, temperature=0, google_api_key=settings.GEMINI_API_KEY)
        self.embeddings = ToolEmbeddings()
        self.system_prompt = SystemMessage(content="""
            You are the Documentation Assistant. 
            Your job is to answer the user's question using ONLY the retrieved context. 
            If the context does not contain the answer, say "I cannot find the answer in the documentation."
            Always cite the source of your information if available.
        """)
        
    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        active_domain = state.get("active_domain")
        
        logger.info(f"RAG Agent activated for query: {user_query[:50]}...")
        query_vector = await self.embeddings.aembed_query(user_query)
        # 2. Retrieve relevant tools to answer documentation questions
        tools = await pg_registry.hybrid_search(
            query=user_query,
            query_embedding=query_vector,
            top_k=5,
            domain_filter=active_domain
        )
        
        if not tools:
            context = "No relevant tool documentation found."
        else:
            context_parts = []
            for t in tools:
                m = t["metadata"]
                context_parts.append(f"Tool: {m['name']}\nDescription: {m['description']}\nDomain: {m['domain']}")
            context = "\n\n---\n\n".join(context_parts)
            
        context_msg = SystemMessage(content=f"RELEVANT SYSTEM CAPABILITIES (TOOLS):\n\n{context}")
        
        response = await self.llm.ainvoke([self.system_prompt, context_msg] + list(messages))
        return {"messages": [response]}

rag_node = RAGAgent()
