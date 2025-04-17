"""ReAct Agent Configuration."""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ReActAgentConfig:
    """Configuration for ReAct agent.
    
    Attributes:
        max_iterations: Maximum number of reasoning-action cycles
        verbose: Whether to print detailed execution logs
        handle_parsing_errors: Whether to handle tool parsing errors gracefully
        allowed_tools: List of tool names that the agent can use
        system_prompt: Custom system prompt for the agent
    """
    
    max_iterations: int = 10
    verbose: bool = True
    handle_parsing_errors: bool = True
    allowed_tools: Optional[List[str]] = None
    system_prompt: str = """You are a helpful AI assistant that can perform tasks using available tools.
    Think step by step and explain your reasoning before taking actions.
    If you're not sure about something, ask for clarification.
    """ 