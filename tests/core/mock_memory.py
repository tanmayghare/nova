"""Mock memory for testing."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

class MockMemory(BaseMemory):
    """Mock memory for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mock memory."""
        super().__init__()
        self.config = config or {}
        self.messages: List[BaseMessage] = []
        
        # Mock methods
        self.add_message = AsyncMock()
        self.clear = AsyncMock()
        self.get_messages = AsyncMock(return_value=[])
        
    def add_message(self, message: BaseMessage) -> None:
        """Mock add_message method."""
        self.messages.append(message)
        
    def clear(self) -> None:
        """Mock clear method."""
        self.messages.clear()
        
    def get_messages(self) -> List[BaseMessage]:
        """Mock get_messages method."""
        return self.messages
        
    def get_context(self, task: str) -> str:
        """Mock get_context method."""
        return "Mock context"
        
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mock load_memory_variables method."""
        return {"history": self.messages}
        
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Mock save_context method."""
        if "input" in inputs:
            self.messages.append(HumanMessage(content=inputs["input"]))
        if "output" in outputs:
            self.messages.append(AIMessage(content=outputs["output"])) 