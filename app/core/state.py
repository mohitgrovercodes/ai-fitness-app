from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Union, Optional
from langchain_core.messages import BaseMessage
from app.schema.orchestration import UserContext

def merge_messages(left: list, right: Any) -> list:
    """Reducer that appends messages by default, but allows full list replacement for memory pruning."""
    if isinstance(right, dict) and right.get("type") == "replace":
        return right.get("messages", [])
    if isinstance(right, list):
        return left + right
    return left + [right]

def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer to safely merge dictionaries from parallel agents."""
    merged = left.copy()
    merged.update(right)
    return merged

class AgentState(TypedDict):
    """
    Production-grade state management.
    """
    messages: Annotated[Sequence[BaseMessage], merge_messages]
    conversation_summary: str  # Stores the rolling summary of past messages
    image_bytes: Optional[bytes]
    
    # Validated User Context
    user_context: UserContext
    
    # Orchestration State
    intent: List[str] # Changed to List for parallel tasks
    is_fitness_domain: bool
    next_node: Union[str, List[str]] # Supports parallel routing
    
    # Internal Specialist Data - safely handles parallel updates
    specialist_results: Annotated[Dict[str, Any], merge_dicts]
    
    # Safety
    is_safe: bool
    safety_reason: Optional[str]
    safety_response: Optional[str] # Added for the polite rejection message
