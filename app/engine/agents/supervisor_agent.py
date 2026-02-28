from typing import Dict, Any, Literal
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.engine.state import AgentState
from app.engine.routing.filters import ToolFilters

logger = logging.getLogger(__name__)

class SupervisorDecision(BaseModel):
    user_intent: str = Field(
        ..., 
        description="A concise summary of what the user is explicitly trying to achieve in this turn."
    )
    next_agent: Literal["action_agent", "rag_agent", "system_agent", "FINISH"] = Field(
        ...,
        description="The agent that should handle this intent. Action -> executing API calls. RAG -> answering 'how-to' questions from docs. System -> debugging or system logs. FINISH -> task is complete."
    )
    is_clarification_needed: bool = Field(
        default=False,
        description="True if the user's intent is too ambiguous to proceed without asking a clarifying question."
    )

class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)        
        self.structured_llm = self.llm.with_structured_output(SupervisorDecision)
        self.system_prompt = SystemMessage(
            content='''
                You are the Chief Routing Supervisor for a highly scalable API Agentic System.
            
                Your job is to read the conversation history and determine WHERE to route the user's request next.
                - Route to 'action_agent' if the user wants to PERFORM an action OR FETCH DATA (e.g., send an invoice, create a payment, fetch sales volume, check balance, or retrieve specific records).
                - Route to 'rag_agent' ONLY if the user is asking a general "How to" question about policies or documentation (e.g., "How do I handle a dispute?").
                - Route to 'system_agent' if the user is asking about the system itself (e.g., "what tools do you have?").
                - Route to 'FINISH' if the user's request has been fully satisfied OR if the assistant has provided a response that requires the user to give more information.
            
                Never attempt to answer the user directly. ALWAYS output the routing JSON.
            ''')

    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        
        prompt = [self.system_prompt] + list(messages)
        
        # TOKEN OPTIMIZATION: If the last message is an assistant response WITHOUT tool calls, 
        # it means the agent already replied to the user. We must FINISH.
        if messages:
            last_msg = messages[-1]
            from langchain_core.messages import AIMessage
            if isinstance(last_msg, AIMessage) and not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                logger.info("Shortcut: Assistant already replied. Routing to FINISH.")
                return {
                    "user_intent": state.get("user_intent", "continue"),
                    "next_agent": "FINISH",
                    "active_domain": state.get("active_domain", "payments"),
                    "tool_error": None 
                }

        decision: SupervisorDecision = await self.structured_llm.ainvoke(prompt)
        
        if not decision:
            logger.error("Supervisor failed to produce a decision.")
            return {"next_agent": "FINISH"}

        logger.info(f"Supervisor Decision: Intent='{decision.user_intent}' -> Route='{decision.next_agent}'")
        
        # We respect the domain set by the domain_router node if it exists
        domain = state.get("active_domain", "payments")
        
        return {
            "user_intent": decision.user_intent,
            "next_agent": decision.next_agent,
            "active_domain": domain,
            "tool_error": None 
        }


supervisor_node = SupervisorAgent()
