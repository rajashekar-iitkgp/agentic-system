from typing import Dict, Any, Literal, List, Optional
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.engine.state import AgentState
from app.engine.routing.filters import ToolFilters

logger = logging.getLogger(__name__)

class StepSpec(BaseModel):
    id: str = Field(..., description="Stable identifier for this logical step (e.g., 'step_1').")
    description: str = Field(..., description="Natural-language description of what this step should accomplish.")
    required_tool: Optional[str] = Field(
        default=None,
        description="If known, the primary tool name expected to execute this step.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of prerequisite steps whose results this step depends on.",
    )


class SupervisorDecision(BaseModel):
    user_intents: List[str] = Field(
        ...,
        description=(
            "A list of distinct sub-queries or actions broken down from the user's message. "
            "For complex multi-step requests, list each step as a separate string. "
            "For simple requests, return a list of one string."
        ),
    )
    next_agent: Literal["action_agent", "rag_agent", "system_agent", "FINISH"] = Field(
        ...,
        description=(
            "The agent that should handle this intent. Action -> executing API calls. "
            "RAG -> answering 'how-to' questions from docs. System -> debugging or system logs. "
            "FINISH -> task is complete."
        ),
    )
    is_clarification_needed: bool = Field(
        default=False,
        description="True if the user's intent is too ambiguous to proceed without asking a clarifying question.",
    )
    steps: List[StepSpec] = Field(
        default_factory=list,
        description=(
            "Optional explicit multi-step plan. Each element is a logical step which may depend "
            "on outputs of previous steps."
        ),
    )
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional confidence scores for routing or key decisions (e.g., per domain or per intent).",
    )

def _infer_domain_from_messages(messages) -> str:
    try:
        from langchain_core.messages import HumanMessage
    except Exception:
        HumanMessage = None

    latest_text = ""
    if messages:
        for m in reversed(messages):
            if HumanMessage is not None and isinstance(m, HumanMessage):
                content = getattr(m, "content", None)
                if isinstance(content, str):
                    latest_text = content
                    break
            else:
                content = getattr(m, "content", None)
                if isinstance(content, str):
                    latest_text = content
                    break

    text = latest_text.lower()

    if any(w in text for w in ["dispute", "chargeback", "case"]):
        return "disputes"
    if any(w in text for w in ["sales", "volume", "report", "revenue"]):
        return "reporting"
    return "payments"


class SupervisorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            google_api_key=settings.GEMINI_API_KEY,
        )
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
        inferred_domain = _infer_domain_from_messages(messages)

        prompt = [self.system_prompt] + list(messages)
        
        if messages:
            last_msg = messages[-1]
            from langchain_core.messages import AIMessage, ToolMessage

            if isinstance(last_msg, AIMessage) and not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                logger.info("Shortcut: Assistant already replied. Routing to FINISH.")
                return {
                    "user_intents": state.get("user_intents", ["continue"]),
                    "next_agent": "FINISH",
                    "active_domain": inferred_domain or "payments",
                    "tool_error": None,
                }

            if isinstance(last_msg, ToolMessage):
                logger.info("Tool execution finished. Routing to FINISH.")
                return {
                    "user_intents": state.get("user_intents", ["continue"]),
                    "next_agent": "FINISH",
                    "active_domain": inferred_domain or "payments",
                    "tool_error": None,
                }

        try:
            decision: SupervisorDecision = await self.structured_llm.ainvoke(prompt)
            if not decision.user_intents:
                 decision.user_intents = state.get("user_intents") or [" "]
        except Exception as e:
            logger.error(f"Supervisor LLM failed: {e}. Falling back to state intent.")
            err_str = str(e).lower()

            fallback = {
                "user_intents": state.get("user_intents") or [" "],
                "active_domain": inferred_domain or "payments",
                "tool_error": None,
            }

            # Treat clear quota / rate-limit signals from the LLM provider as a quota exhaustion
            # flag so downstream agents can avoid additional LLM calls and return a clear message.
            if any(s in err_str for s in ["insufficient_quota", "quota", "rate limit", "resource_exhausted", "429"]):
                logger.warning(
                    "Detected LLM quota/rate-limit error in Supervisor. "
                    "Routing to action_agent with openai_quota_error flag."
                )
                fallback.update(
                    {
                        "next_agent": "action_agent",
                        "openai_quota_error": True,
                    }
                )
            else:
                fallback.update({"next_agent": "FINISH"})
            return fallback
        
        logger.info(f"Supervisor Decision: Intents='{decision.user_intents}' -> Route='{decision.next_agent}'")

        # Map structured decision into the graph state. For backwards compatibility we still
        # expose user_intents, but we also attach the richer planning fields when present.
        next_iteration = state.get("iteration_count", 0) + 1
        max_iterations = state.get("max_iterations", 6)

        # Hard cap to avoid any possibility of infinite loops even if upstream logic changes.
        if next_iteration > max_iterations:
            logger.info(
                f"Iteration cap reached ({max_iterations}). Forcing FINISH to avoid long-running loops."
            )
            return {
                "user_intents": state.get("user_intents") or decision.user_intents,
                "next_agent": "FINISH",
                "active_domain": inferred_domain or "payments",
                "tool_error": None,
                "steps": [s.model_dump() for s in decision.steps] if decision.steps else state.get("steps", []),
                "confidence_scores": decision.confidence_scores or state.get("confidence_scores", {}),
                "iteration_count": next_iteration,
                "max_iterations": max_iterations,
            }

        return {
            "user_intents": decision.user_intents,
            "next_agent": decision.next_agent,
            "active_domain": inferred_domain or "payments",
            "tool_error": None,
            "steps": [s.model_dump() for s in decision.steps] if decision.steps else state.get("steps", []),
            "confidence_scores": decision.confidence_scores or state.get("confidence_scores", {}),
            "iteration_count": next_iteration,
            "max_iterations": max_iterations,
        }


supervisor_node = SupervisorAgent()
