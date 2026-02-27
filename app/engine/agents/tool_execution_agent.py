from typing import Dict, Any, List, Callable
import logging
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool

from app.core.config import settings
from app.engine.state import AgentState

logger = logging.getLogger(__name__)

class ToolExecutionAgent:
    """
    Sub-agent responsible for executing actions.
    It receives a dynamically filtered list of tools (ActiveToolDict) from the state,
    binds ONLY those tools to the LLM, and executes the sequence.
    """
    def __init__(self, all_tool_implementations: Dict[str, Callable]):
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        self.all_tool_implementations = all_tool_implementations

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        The agent node for tool execution.
        """
        messages = state.get("messages", [])
        retrieved_tools = state.get("retrieved_tools", [])
        
        if not retrieved_tools:
            logger.warning("No tools retrieved for ToolExecutionAgent. Falling back.")
            return {"messages": [AIMessage(content="I'm sorry, I couldn't find any tools to help with that request.")]}

        logger.info(f"ToolExecutionAgent binding {len(retrieved_tools)} tools.")
        
        # 1. Dynamically construct Langchain Tools from the state schemas and the real python functions
        active_langchain_tools = []
        for t in retrieved_tools:
            tool_name = t["name"]
            if tool_name in self.all_tool_implementations:
                # We create a StructuredTool dynamically
                # Note: In production, you would convert the t["input_schema"] into a Pydantic model
                # or use the Langchain format. For this example, we assume the python function signature matches.
                func = self.all_tool_implementations[tool_name]
                active_langchain_tools.append(func)
            else:
                logger.error(f"Implementation for tool {tool_name} not found!")

        # 2. Bind the active tools to the LLM
        if active_langchain_tools:
            llm_with_tools = self.llm.bind_tools(active_langchain_tools)
        else:
            llm_with_tools = self.llm
            
        system_prompt = SystemMessage(content="You are a helpful assistant executing actions. Use the provided tools to fulfill the user's request. If you need more information, ask the user.")
        
        # 3. Invoke LLM
        # We include the system prompt to guide the action agent
        response = await llm_with_tools.ainvoke([system_prompt] + list(messages))
        
        output_messages = [response]
        
        # 4. If the LLM decided to call tools, execute them immediately
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]
                
                logger.info(f"Executing tool {tool_name} with args {args}")
                
                if tool_name in self.all_tool_implementations:
                    try:
                        func = self.all_tool_implementations[tool_name]
                        # Execute the async tool function
                        result = await func(**args)
                        
                        # Append the result as a ToolMessage
                        output_messages.append(ToolMessage(
                            tool_call_id=call_id,
                            content=str(result)
                        ))
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        output_messages.append(ToolMessage(
                            tool_call_id=call_id,
                            content=f"Error: {str(e)}"
                        ))
                else:
                    output_messages.append(ToolMessage(
                        tool_call_id=call_id,
                        content=f"Error: Tool '{tool_name}' implementation not found."
                    ))

        return {"messages": output_messages}

# In a full setup, we would initialize this with the mapped python functions
# tool_execution_node = ToolExecutionAgent(all_implementations)
