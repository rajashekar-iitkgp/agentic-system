import logging
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, Callable

from app.engine.state import AgentState
from app.engine.agents.supervisor_agent import supervisor_node
from app.engine.agents.tool_execution_agent import ToolExecutionAgent
from app.engine.agents.rag_agent import rag_node
from app.engine.agents.system_query_agent import system_node
from app.engine.routing.semantic_router import router
from app.engine.routing.domain_router import domain_router
from app.core.config import settings

logger = logging.getLogger(__name__)

def _get_checkpointer():
    """Get a checkpointer for the LangGraph."""
    from langgraph.checkpoint.memory import MemorySaver
    logger.info("Using MemorySaver for graph compilation.")
    return MemorySaver(), None

def build_graph(all_tool_implementations: Dict[str, Callable]) -> Any:
    workflow = StateGraph(AgentState)
    action_node = ToolExecutionAgent(all_tool_implementations)

    async def supervisor(state: AgentState):
        result = await supervisor_node.run(state)
        return {
            **result, 
            "trace_id": state.get("trace_id"), 
            "request_metadata": state.get("request_metadata"),
            "active_domain": state.get("active_domain")
        }

    async def domain_routing_node(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"active_domain": "payments"}
        
        last_message = messages[-1].content
        domain = await domain_router.classify(last_message)
        return {"active_domain": domain}
        
    async def router_node(state: AgentState):
        intent = state.get("user_intent")
        domain = state.get("active_domain")
        tools = await router.retrieve_tools_for_intent(intent, domain_filter=domain, k=5)
        return {"retrieved_tools": tools}

    async def action(state: AgentState):
        return await action_node.run(state)

    async def rag(state: AgentState):
        return await rag_node.run(state)

    async def system(state: AgentState):
        return await system_node.run(state)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("domain_router", domain_routing_node)
    workflow.add_node("semantic_router", router_node)
    workflow.add_node("action_agent", action)
    workflow.add_node("rag_agent", rag)
    workflow.add_node("system_agent", system)
    
    workflow.add_edge(START, "domain_router")
    workflow.add_edge("domain_router", "supervisor")

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

    workflow.add_edge("semantic_router", "action_agent")
    workflow.add_edge("action_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("system_agent", "supervisor")

    checkpointer, async_pool = _get_checkpointer()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    compiled_graph._async_pool = async_pool
    logger.info("Agentic Graph compiled with MemorySaver.")
    return compiled_graph
