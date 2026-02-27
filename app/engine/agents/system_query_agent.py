import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.engine.state import AgentState

logger = logging.getLogger(__name__)


class SystemQueryAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=settings.GEMINI_API_KEY)
        self.system_prompt = SystemMessage(content="""
        You are the System Status Assistant.
        You have access to the system registry and metadata logs to answer user questions about their tool usage and system capabilities.
        """)
        
    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        retrieved_tools = state.get("retrieved_tools", [])
        logger.info("System Query Agent activated.")
        tool_names = [t["name"] for t in retrieved_tools]
        context = f"Internal Debug Info - Retrieved tools for this query: {tool_names}"
        context_msg = SystemMessage(content=context)
        response = await self.llm.ainvoke([self.system_prompt, context_msg] + list(messages))
        return {"messages": [response]}

system_node = SystemQueryAgent()
