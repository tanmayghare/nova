"""Mock LLM for testing."""

from typing import List, Optional, Union
from unittest.mock import AsyncMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class MockLLM(BaseChatModel):
    """Mock LLM for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize the mock LLM.
        
        Args:
            responses: List of responses to return in sequence
        """
        super().__init__()
        self.responses = responses or ["Mock response"]
        self.current_response = 0
        self.invoke = AsyncMock(side_effect=self._invoke)
        self.ainvoke = AsyncMock(side_effect=self._invoke)
        
    async def _invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Mock invoke method."""
        response = self.responses[self.current_response % len(self.responses)]
        self.current_response += 1
        return AIMessage(content=response)
    
    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Mock ainvoke method."""
        return await self._invoke(messages)
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Mock invoke method."""
        return self._invoke(messages)
    
    def _generate(self, messages: List[BaseMessage]) -> AIMessage:
        """Mock _generate method."""
        return self._invoke(messages)
    
    async def _agenerate(self, messages: List[BaseMessage]) -> AIMessage:
        """Mock _agenerate method."""
        return await self._invoke(messages)
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type."""
        return "mock" 