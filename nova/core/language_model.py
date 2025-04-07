from typing import Dict, Protocol

class LanguageModel(Protocol):
    """Protocol defining the interface for language models."""
    
    def generate_plan(self, task: str, context: Dict[str, str]) -> Dict[str, str]:
        """Generate a plan for executing a task.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            
        Returns:
            A structured plan as a dictionary
        """
        ...
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response to a prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated response text
        """
        ... 