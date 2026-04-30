from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Union
from langchain_core.messages import BaseMessage
from app.schema.orchestration import UserContext

def merge_messages(left: list, right: list) -> list:
    return left + right

class AgentState(TypedDict):
    """
    Production-grade state management.
    """
    messages: Annotated[Sequence[BaseMessage], merge_messages]
    
    # Validated User Context
    user_context: UserContext
    
    # Orchestration State
    intent: str
    is_fitness_domain: bool
    next_node: Union[str, List[str]] # Supports parallel routing
    
    # Internal Specialist Data
    specialist_results: Dict[str, Any]
    is_safe: bool
    safety_reason: Optional[str]
