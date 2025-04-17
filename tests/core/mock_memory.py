"""Mock memory for testing."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from nova.core.memory import Memory

class MockMemory(Memory):
    """Mock memory for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mock memory."""
        super().__init__(config or {})
        
        # Mock methods
        self.add = AsyncMock()
        self.get = AsyncMock(return_value=[])
        self.clear = AsyncMock()
        self.get_context = AsyncMock(return_value="Mock context")
        
    async def add(self, item: Dict[str, Any]) -> None:
        """Mock add method."""
        await self.add(item)
        
    async def get(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock get method."""
        return await self.get(limit)
        
    async def clear(self) -> None:
        """Mock clear method."""
        await self.clear()
        
    async def get_context(self, task: str) -> str:
        """Mock get_context method."""
        return await self.get_context(task) 