from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

def merge_messages(left: list, right: list) -> list:
    return left + right

class AgentState(TypedDict):
    """
    The central state for the Orchestrator.
    """
    messages: Annotated[Sequence[BaseMessage], merge_messages]
    
    # Routing & Control
    next_node: str
    intent: Optional[str]
    is_fitness_domain: bool
    
    # Shared Data
    specialist_output: Dict[str, Any]
    user_context: Dict[str, Any] # Simple context for now (goals, etc)
