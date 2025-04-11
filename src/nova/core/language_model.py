from typing import Dict, List, Any, Protocol

class LanguageModel(Protocol):
    """Protocol defining the interface for language models."""
    
    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            
        Returns:
            A list of action steps
        """
        ...
        
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results.
        
        Args:
            task: The original task
            plan: The executed plan steps
            context: Additional context
            
        Returns:
            The generated response text
        """
        ... 