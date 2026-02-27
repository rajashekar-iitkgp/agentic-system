import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.engine.state import AgentState

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

class SystemQueryAgent:
    """
    Sub-agent dedicated to answering metadata questions like 
    "What tools are available?" or "What's the status of my request?"
    """
    def __init__(self):
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY") 
        )
        self.system_prompt = SystemMessage(content="""
        You are the System Status Assistant.
        You have access to the system registry and metadata logs to answer user questions about their tool usage and system capabilities.
        """)
        
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes a query against the system state or Tool registry.
        """
        messages = state.get("messages", [])
        retrieved_tools = state.get("retrieved_tools", [])
        
        logger.info("System Query Agent activated.")
        
        # If the user asked "What invoice tools do you have?", the Semantic Router would have 
        # populated `retrieved_tools` with internal invoice manipulation tools.
        
        tool_names = [t["name"] for t in retrieved_tools]
        context = f"Internal Debug Info - Retrieved tools for this query: {tool_names}"
        context_msg = SystemMessage(content=context)
        
        response = await self.llm.ainvoke([self.system_prompt, context_msg] + list(messages))
        
        return {"messages": [response]}

system_node = SystemQueryAgent()
