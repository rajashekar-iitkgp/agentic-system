from typing import Dict, Any, List, Callable
import logging
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool

from app.core.config import settings
from app.engine.state import AgentState

from app.engine.execution.gateway import execution_gateway

logger = logging.getLogger(__name__)

class ToolExecutionAgent:
    def __init__(self, all_tool_implementations: Dict[str, Callable]):
        self.llm = ChatGoogleGenerativeAI(model=settings.GENAI_MODEL,temperature=0,google_api_key=settings.GEMINI_API_KEY,)
        self.all_tool_implementations = all_tool_implementations

    async def run(self, state: AgentState) -> Dict[str, Any]:
        if state.get("openai_quota_error"):
            logger.warning("LLM quota/rate-limit flag detected. Skipping tool LLM execution.")
            notice = AIMessage(
                content=(
                    "I identified the relevant tools for your request, but cannot execute them right now "
                    "because the underlying LLM API quota or rate limit has been exceeded. "
                    "Please try again later or update the API billing/quota settings."
                )
            )
            return {"messages": [notice]}

        messages = state.get("messages", [])
        retrieved_tools = state.get("retrieved_tools", [])
        logger.info(f"ToolExecutionAgent binding {len(retrieved_tools)} tools.")

        tool_schema_map = {}
        active_langchain_tools: List[Callable] = []
        for t in retrieved_tools:
            tool_name = t["name"]
            if tool_name in self.all_tool_implementations:
                func = self.all_tool_implementations[tool_name]
                active_langchain_tools.append(func)
                if hasattr(func, "args_schema"):
                    tool_schema_map[tool_name] = func.args_schema
            else:
                logger.error(f"Implementation for tool {tool_name} not found!")

        if active_langchain_tools:
            llm_with_tools = self.llm.bind_tools(active_langchain_tools)
        else:
            llm_with_tools = self.llm
            
        must_use_tools = bool(retrieved_tools)
        system_prompt_text = (
            "You are a strict action execution engine. "
            "If any tools are available, you MUST respond using tool calls only and NEVER answer directly. "
            "Execute tools immediately when required parameters are present. "
            "Do not request extra fields beyond the tool schema. "
            "Ask the user only when a required parameter is missing."
        )
        system_prompt = SystemMessage(content=system_prompt_text)
        
        current_messages = [system_prompt] + list(messages)
        output_messages = []
        MAX_ATTEMPTS = 2
        step_results = state.get("step_results", {})
        
        for attempt in range(MAX_ATTEMPTS):
            response = await llm_with_tools.ainvoke(current_messages)
            current_messages.append(response)
            
            if attempt == 0:
                output_messages = [response]
            else:
                output_messages.append(response)
                
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                break
                
            all_tools_successful = True

            import asyncio

            async def execute_tool(tool_call):
                tool_name = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]
                
                try:
                    schema = tool_schema_map.get(tool_name)
                    execution_gateway.validate_tool_call(tool_name, args, schema, state)
                    
                    logger.info(f"Executing tool {tool_name} with args {args}")
                    if tool_name in self.all_tool_implementations:
                        func = self.all_tool_implementations[tool_name]
                        result = await func(**args)
                        # Record a simple step result keyed by tool_call id to support
                        # downstream multi-step reasoning without changing existing flows.
                        step_results[call_id] = {
                            "step_id": call_id,
                            "status": "success",
                            "tool_name": tool_name,
                            "output": {"raw": str(result)},
                            "error": None,
                        }
                        return ToolMessage(tool_call_id=call_id, content=str(result)), True
                    else:
                        return ToolMessage(tool_call_id=call_id, content=f"Error: Tool '{tool_name}' implementation not found."), False
                
                except ValueError as ve:
                    logger.warning(f"Gateway Blocked Execution: {ve}")
                    step_results[call_id] = {
                        "step_id": call_id,
                        "status": "failed",
                        "tool_name": tool_name,
                        "output": None,
                        "error": str(ve),
                    }
                    return ToolMessage(tool_call_id=call_id, content=str(ve)), False
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    step_results[call_id] = {
                        "step_id": call_id,
                        "status": "failed",
                        "tool_name": tool_name,
                        "output": None,
                        "error": str(e),
                    }
                    return ToolMessage(tool_call_id=call_id, content=f"Error: {str(e)}"), False

            tool_tasks = [execute_tool(t) for t in response.tool_calls]
            execution_results = await asyncio.gather(*tool_tasks)
            
            for tool_msg, success in execution_results:
                current_messages.append(tool_msg)
                output_messages.append(tool_msg)
                if not success:
                    all_tools_successful = False
            
            if all_tools_successful:
                break

        if output_messages and isinstance(output_messages[-1], ToolMessage) and "Error" in str(output_messages[-1].content):
            final_abort_msg = AIMessage(content=f"Tool execution failed: {output_messages[-1].content}")
            output_messages.append(final_abort_msg)

        return {"messages": output_messages, "step_results": step_results}
