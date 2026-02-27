from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class ActiveToolDict(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: Optional[str]
    user_intent: Optional[str]
    active_domain: Optional[str]
    retrieved_tools: Annotated[List[ActiveToolDict], lambda a, b: b]
    tool_error: Optional[str]