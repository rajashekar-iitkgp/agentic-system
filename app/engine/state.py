from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class ActiveToolDict(TypedDict):
    """
    Represents a tool that has been dynamically retrieved from the Vector Store
    and is currently available for the agent to use in this specific turn.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]

class AgentState(TypedDict):
    """
    The core state of the LangGraph execution.
    This persists across asynchronous node boundaries.
    """
    # Conversation History. The `add_messages` reducer appends new messages.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Internal routing decision made by the Supervisor
    next_agent: Optional[str]
    
    # The current user intent extracted implicitly 
    user_intent: Optional[str]
    
    # The domain (e.g., 'payments', 'system', 'rag') the supervisor assigned
    active_domain: Optional[str]
    
    # The TOP K dynamically retrieved tools relevant ONLY to the current query
    # We explicitly overwrite this list each routing phase rather than appending.
    retrieved_tools: Annotated[List[ActiveToolDict], lambda a, b: b]
    
    # If a specific error occurred during tool execution that the agent needs to self-correct
    tool_error: Optional[str]
