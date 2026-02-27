from typing import Dict, Any, List, Callable
import logging
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool

from app.core.config import settings
from app.engine.state import AgentState

logger = logging.getLogger(__name__)

class ToolExecutionAgent:
    def __init__(self, all_tool_implementations: Dict[str, Callable]):
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=settings.GEMINI_API_KEY)
        self.all_tool_implementations = all_tool_implementations

    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        retrieved_tools = state.get("retrieved_tools", [])
        logger.info(f"ToolExecutionAgent binding {len(retrieved_tools)} tools.")
        
        active_langchain_tools = []
        for t in retrieved_tools:
            tool_name = t["name"]
            if tool_name in self.all_tool_implementations:
                func = self.all_tool_implementations[tool_name]
                active_langchain_tools.append(func)
            else:
                logger.error(f"Implementation for tool {tool_name} not found!")

        if active_langchain_tools:
            llm_with_tools = self.llm.bind_tools(active_langchain_tools)
        else:
            llm_with_tools = self.llm
            
        system_prompt = SystemMessage(content="You are a helpful assistant executing actions. Use the provided tools to fulfill the user's request. If you need more information, ask the user.")
        response = await llm_with_tools.ainvoke([system_prompt] + list(messages))
        output_messages = [response]
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]      
                logger.info(f"Executing tool {tool_name} with args {args}")
                
                if tool_name in self.all_tool_implementations:
                    try:
                        func = self.all_tool_implementations[tool_name]
                        result = await func(**args)
                        output_messages.append(ToolMessage(tool_call_id=call_id, content=str(result)))
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        output_messages.append(ToolMessage(tool_call_id=call_id, content=f"Error: {str(e)}"))
                else:
                    output_messages.append(ToolMessage(tool_call_id=call_id, content=f"Error: Tool '{tool_name}' implementation not found."))

        return {"messages": output_messages}