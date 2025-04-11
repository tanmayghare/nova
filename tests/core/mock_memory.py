"""Mock memory implementation for testing."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class MockMemory:
    """Mock memory implementation for testing."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the mock memory."""
        self.max_size = max_size
        self._memories = []
        self._last_cleanup = datetime.now()
        
    async def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a memory entry."""
        # Clean up old memories if needed
        await self._cleanup()
        
        # Add new memory
        memory = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        self._memories.append(memory)
        
        # Trim if over max size
        if len(self._memories) > self.max_size:
            self._memories = self._memories[-self.max_size:]
            
    async def get_memories(
        self,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get memories."""
        # Clean up old memories
        await self._cleanup()
        
        # Filter by timestamp if specified
        memories = self._memories
        if since:
            memories = [
                m for m in memories
                if m["timestamp"] >= since
            ]
            
        # Apply limit if specified
        if limit:
            memories = memories[-limit:]
            
        return memories
        
    async def get_context(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for a query."""
        # Clean up old memories
        await self._cleanup()
        
        # Simple keyword matching for testing
        relevant_memories = [
            m for m in self._memories
            if any(
                word.lower() in m["content"].lower()
                for word in query.split()
            )
        ]
        
        # Apply limit if specified
        if limit:
            relevant_memories = relevant_memories[-limit:]
            
        return relevant_memories
        
    async def _cleanup(self) -> None:
        """Clean up old memories."""
        now = datetime.now()
        if now - self._last_cleanup > timedelta(hours=1):
            self._memories = [
                m for m in self._memories
                if now - m["timestamp"] < timedelta(days=7)
            ]
            self._last_cleanup = now
            
    async def clear(self) -> None:
        """Clear all memories."""
        self._memories = []
        self._last_cleanup = datetime.now() 