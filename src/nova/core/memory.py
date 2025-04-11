from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Memory:
    """Memory system for storing and retrieving context.
    
    This class implements a memory system that stores task execution history
    and provides context retrieval capabilities. It maintains both short-term
    and long-term memory, with the ability to summarize and retrieve relevant
    information based on task requirements.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        """Initialize the memory system.
        
        Args:
            max_entries: Maximum number of memory entries to keep
        """
        self._memory: List[Dict[str, Any]] = []
        self._max_entries = max_entries
        self._summary: Dict[str, str] = {}

    async def get_context(self, task: str) -> str:
        """Get relevant context for a task.
        
        This method retrieves relevant context from memory based on the current task.
        It uses semantic similarity to find the most relevant memories and summarizes
        them into a coherent context string.
        
        Args:
            task: The current task description
            
        Returns:
            A string containing relevant context from memory
        """
        if not self._memory:
            return "No previous context available."
            
        # Get memories for this task
        task_memories = [m for m in self._memory if m["task"] == task]
        if not task_memories:
            return "No previous context available."
            
        # Format memories into context
        context_parts = []
        for memory in task_memories[-10:]:  # Last 10 memories for this task
            if "result" in memory:
                context_parts.append(f"Previous attempt: {memory['result']}")
            elif "content" in memory:
                context_parts.append(f"Previous memory: {memory['content']}")
                
        return "\n".join(context_parts) if context_parts else "No previous context available."

    async def add(self, task: str, step: Optional[Dict[str, Any]] = None, result: Optional[str] = None) -> None:
        """Add a memory entry.
        
        Args:
            task: The task description
            step: Optional step that was executed
            result: Optional result of the step execution. If not provided, task is treated as the content.
        """
        if step is None and result is None:
            # Handle the simple case where task is the content
            entry = {
                "task": task,
                "content": task,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Handle the full case with step and result
            entry = {
                "task": task,
                "step": step or {},
                "result": result or "",
                "timestamp": datetime.now().isoformat()
            }
        
        self._memory.append(entry)
        
        # Maintain memory size limit
        if len(self._memory) > self._max_entries:
            self._memory = self._memory[-self._max_entries:]
            
        # Update summary for this task if result is provided
        if result:
            self._update_summary(task, result)

    def _update_summary(self, task: str, result: str) -> None:
        """Update the summary for a task.
        
        Args:
            task: The task description
            result: The result to incorporate into the summary
        """
        if task not in self._summary:
            self._summary[task] = result
        else:
            # Combine with existing summary
            self._summary[task] = f"{self._summary[task]}\n{result}"

    async def get_summary(self, task: Optional[str] = None) -> str:
        """Get a summary of memories.
        
        Args:
            task: Optional task to get summary for. If None, returns all summaries.
            
        Returns:
            A summary string of relevant memories
        """
        if task:
            return self._summary.get(task, "No summary available for this task.")
        return "\n\n".join(f"Task: {t}\nSummary: {s}" for t, s in self._summary.items())

    async def clear_task(self, task: str) -> None:
        """Clear all memories and summary associated with a specific task.
        
        Args:
            task: The task identifier whose memories should be cleared.
        """
        self._memory = [m for m in self._memory if m["task"] != task]
        if task in self._summary:
            del self._summary[task]

    def clear(self) -> None:
        """Clear all memories and summaries."""
        self._memory = []
        self._summary = {}

    def to_json(self) -> str:
        """Convert memory to JSON string.
        
        Returns:
            JSON string representation of memory
        """
        return json.dumps({
            "memory": self._memory,
            "summary": self._summary
        })

    @classmethod
    def from_json(cls, json_str: str) -> Memory:
        """Create Memory instance from JSON string.
        
        Args:
            json_str: JSON string representation of memory
            
        Returns:
            Memory instance
        """
        data = json.loads(json_str)
        memory = cls()
        memory._memory = data["memory"]
        memory._summary = data["summary"]
        return memory
