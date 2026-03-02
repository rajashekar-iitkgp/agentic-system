import logging
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, Callable

from app.engine.state import AgentState
from app.engine.agents.supervisor_agent import supervisor_node
from app.engine.agents.tool_execution_agent import ToolExecutionAgent
from app.engine.agents.rag_agent import rag_node
from app.engine.agents.system_query_agent import system_node
from app.engine.routing.semantic_router import router
from app.core.config import settings
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
import asyncio

logger = logging.getLogger(__name__)

def _get_checkpointer():
    logger.info("Using MemorySaver for graph compilation.")
    return MemorySaver(), None

def build_graph(all_tool_implementations: Dict[str, Callable]) -> Any:
    workflow = StateGraph(AgentState)
    action_node = ToolExecutionAgent(all_tool_implementations)

    async def supervisor(state: AgentState):
        result = await supervisor_node.run(state)
        # Merge supervisor output back into state, preserving orchestration fields where present.
        return {
            **state,
            **result,
            "tool_error": result.get("tool_error", state.get("tool_error")),
            "requires_reseed": result.get("requires_reseed", state.get("requires_reseed")),
            "trace_id": state.get("trace_id"),
            "request_metadata": state.get("request_metadata"),
            "active_domain": result.get("active_domain", state.get("active_domain")),
        }
        
    async def router_node(state: AgentState):
        intents = state.get("user_intents", [])
        if not intents:
            legacy_intent = state.get("user_intent")
            intents = [legacy_intent] if legacy_intent else [""]
        active_domain = state.get("active_domain") or "payments"
        tasks = [router.retrieve_tools_for_intent(intent, domain_filter=active_domain, k=3) for intent in intents]
        results = await asyncio.gather(*tasks)
        
        all_tools = []  
        for tools in results:
            all_tools.extend(tools)
            
        # Deduplicate tools by name
        unique_tools_map = {t["name"]: t for t in all_tools}
        return {"retrieved_tools": list(unique_tools_map.values())}

    async def action(state: AgentState):
        return await action_node.run(state)

    async def rag(state: AgentState):
        return await rag_node.run(state)

    async def system(state: AgentState):
        return await system_node.run(state)

    async def tool_not_found(state: AgentState):
        return {"messages": [AIMessage(content="I'm sorry, I don't have the capability to perform this action yet. Please contact support or check available commands.")]}

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("semantic_router", router_node)
    workflow.add_node("action_agent", action)
    workflow.add_node("rag_agent", rag)
    workflow.add_node("system_agent", system)
    workflow.add_node("tool_not_found", tool_not_found)
    
    workflow.add_edge(START, "supervisor")

    def route_decision(state: AgentState) -> str:
        decision = state.get("next_agent", "FINISH")
        if decision == "FINISH":
            return END
        if decision == "action_agent":
            return "semantic_router"
        return decision

    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "semantic_router": "semantic_router",
            "rag_agent": "rag_agent",
            "system_agent": "system_agent",
            END: END
        }
    )

    def route_after_search(state: AgentState) -> str:
        tools = state.get("retrieved_tools", [])
        if not tools:
            logger.info("No tools found for intent. Returning 'not found' message.")
            return "tool_not_found"
        return "action_agent"

    workflow.add_conditional_edges("semantic_router", route_after_search, {"action_agent": "action_agent", "tool_not_found": "tool_not_found"})
    workflow.add_edge("tool_not_found", END)
    workflow.add_edge("action_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("system_agent", "supervisor")

    checkpointer, async_pool = _get_checkpointer()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    compiled_graph._async_pool = async_pool
    logger.info("Agentic Graph compiled with MemorySaver.")
    return compiled_graph
