from __future__ import annotations

from typing import Any, Dict, List, Optional


class Memory:
    """Memory system for storing and retrieving context."""

    def __init__(self) -> None:
        """Initialize an empty memory."""
        self._memory: List[Dict[str, Any]] = []

    async def get_context(self, task: str) -> str:
        """Get relevant context for a task."""
        # TODO: Implement actual context retrieval
        return "Context"

    async def add(self, task: str, step: Dict[str, Any], result: str) -> None:
        """Add a memory entry."""
        self._memory.append({
            "task": task,
            "step": step,
            "result": result,
        })

    def clear(self) -> None:
        """Clear all memories."""
        self._memory.clear()
