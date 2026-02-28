from typing import Dict, Any, List, Callable
import logging
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool

from app.core.config import settings
from app.engine.state import AgentState

from app.engine.execution.gateway import execution_gateway

logger = logging.getLogger(__name__)

class ToolExecutionAgent:
    def __init__(self, all_tool_implementations: Dict[str, Callable]):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
        self.all_tool_implementations = all_tool_implementations

    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        retrieved_tools = state.get("retrieved_tools", [])
        logger.info(f"ToolExecutionAgent binding {len(retrieved_tools)} tools.")
        
        # Mapping for validation
        tool_schema_map = {}
        active_langchain_tools = []
        for t in retrieved_tools:
            tool_name = t["name"]
            if tool_name in self.all_tool_implementations:
                func = self.all_tool_implementations[tool_name]
                active_langchain_tools.append(func)
                # If it's a StructuredTool, we can get the schema
                if hasattr(func, "args_schema"):
                    tool_schema_map[tool_name] = func.args_schema
            else:
                logger.error(f"Implementation for tool {tool_name} not found!")

        if active_langchain_tools:
            llm_with_tools = self.llm.bind_tools(active_langchain_tools)
        else:
            llm_with_tools = self.llm
            
        system_prompt = SystemMessage(content="""You are a strict action execution engine. 
        Your goal is to execute tools immediately if you have the required parameters.
        DO NOT ask for additional fields (like email, description, etc.) if they are not explicitly required by the tool's schema.
        If the user's request can be fulfilled by a tool call with the available information, execute it NOW.
        Only ask the user for information if a REQUIRED parameter in the tool schema is missing.""")
        response = await llm_with_tools.ainvoke([system_prompt] + list(messages))
        output_messages = [response]
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]      
                
                try:
                    # VALIDATION GATEWAY
                    schema = tool_schema_map.get(tool_name)
                    execution_gateway.validate_tool_call(tool_name, args, schema, state)
                    
                    logger.info(f"Executing tool {tool_name} with args {args}")
                    if tool_name in self.all_tool_implementations:
                        func = self.all_tool_implementations[tool_name]
                        result = await func(**args)
                        output_messages.append(ToolMessage(tool_call_id=call_id, content=str(result)))
                    else:
                        output_messages.append(ToolMessage(tool_call_id=call_id, content=f"Error: Tool '{tool_name}' implementation not found."))
                
                except ValueError as ve:
                    logger.warning(f"Gateway Blocked Execution: {ve}")
                    output_messages.append(ToolMessage(tool_call_id=call_id, content=f"Policy/Validation Error: {str(ve)}"))
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    output_messages.append(ToolMessage(tool_call_id=call_id, content=f"Error: {str(e)}"))

        return {"messages": output_messages}
