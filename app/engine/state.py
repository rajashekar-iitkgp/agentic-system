from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class ActiveToolDict(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]


class StepResult(TypedDict, total=False):
    step_id: str
    status: str  # "pending" | "success" | "failed"
    tool_name: Optional[str]
    output: Optional[Dict[str, Any]]
    error: Optional[str]


class CompAction(TypedDict, total=False):
    tool_name: str
    args: Dict[str, Any]


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: Optional[str]
    user_intents: List[str]
    active_domain: Optional[str]
    retrieved_tools: Annotated[List[ActiveToolDict], lambda a, b: b]
    tool_error: Optional[str]
    requires_reseed: Optional[bool]
    trace_id: str
    request_metadata: Dict[str, Any]
    openai_quota_error: Optional[bool]

    # v2 orchestration fields (all optional for backward compatibility)
    steps: List[Dict[str, Any]]
    step_results: Dict[str, StepResult]
    completed_steps: List[str]
    pending_compensation: List[CompAction]
    clarification_pending: bool
    confidence_scores: Dict[str, float]
    iteration_count: int
    max_iterations: int
