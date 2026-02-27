import logging
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, Callable

from app.engine.state import AgentState
from app.engine.agents.supervisor_agent import supervisor_node
from app.engine.agents.tool_execution_agent import ToolExecutionAgent
from app.engine.agents.rag_agent import rag_node
from app.engine.agents.system_query_agent import system_node
from app.engine.routing.semantic_router import router

logger = logging.getLogger(__name__)

def build_graph(all_tool_implementations: Dict[str, Callable]) -> Any:
    """
    Compiles the asynchronous LangGraph orchestration pipeline.
    """
    workflow = StateGraph(AgentState)
    
    # Initialize the Action Agent with the concrete tool registry implementations
    action_node = ToolExecutionAgent(all_tool_implementations)

    # ==========================
    # 1. Define the Nodes
    # ==========================
    
    # We wrap the supervisor node to correctly map input/output dictionaries matching the State
    async def supervisor(state: AgentState):
        return await supervisor_node.run(state)
        
    async def router_node(state: AgentState):
        """
        Intermediate node that runs strictly BEFORE the action_agent.
        Retrieves the exact tools needed for the action_agent to succeed.
        """
        intent = state.get("user_intent")
        domain = state.get("active_domain")
        if not intent:
            # Fallback if supervisor failed to extract intent
            return {"retrieved_tools": []}
            
        tools = await router.retrieve_tools_for_intent(intent, domain_filter=domain, k=5)
        return {"retrieved_tools": tools}

    async def action(state: AgentState):
        return await action_node.run(state)

    async def rag(state: AgentState):
        return await rag_node.run(state)

    async def system(state: AgentState):
        return await system_node.run(state)

    # Add Nodes to Graph
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("semantic_router", router_node)
    workflow.add_node("action_agent", action)
    workflow.add_node("rag_agent", rag)
    workflow.add_node("system_agent", system)

    # ==========================
    # 2. Define the Edges
    # ==========================
    
    # The entrypoint is ALWAYS the supervisor to assess the user's input
    workflow.add_edge(START, "supervisor")

    # Conditional Routing Logic
    def route_decision(state: AgentState) -> str:
        decision = state.get("next_agent", "FINISH")
        if decision == "FINISH":
            return END
        # If the choice is action_agent, we MUST route through the semantic_router first!
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

    # Sequence Edges: The semantic router strictly points to the action_agent
    workflow.add_edge("semantic_router", "action_agent")
    
    # All sub-agents loop back to the supervisor to check if further action is needed
    workflow.add_edge("action_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("system_agent", "supervisor")

    # Compile the graph
    # Optional: pass checkpointer here for persistence
    # app = workflow.compile(checkpointer=memory_store)
    
    compiled_graph = workflow.compile()
    logger.info("Async Agentic Graph compiled successfully.")
    
    return compiled_graph
