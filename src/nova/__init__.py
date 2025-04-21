"""Nova - An autonomous web agent framework."""

from .core.browser import Browser
# from .core.config import AgentConfig # AgentConfig is defined in core.agents now
from .browser.config import BrowserConfig
from .core.llm import LLM, LLMConfig # Export LLMConfig too
from .core.memory import Memory
from .core.tools import Tool, ToolRegistry, ToolResult
# Removed imports from deleted agents directory
# from .agents.task.task_agent import TaskAgent
# from .agents.base.base_agent import BaseAgent
from .core.agents import BaseAgent, TaskAgent, AgentConfig # Import core agents and config

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "TaskAgent",
    "AgentConfig", # Added AgentConfig from core.agents
    "Browser",
    "BrowserConfig",
    "LLM",
    "LLMConfig", # Added LLMConfig
    "Memory",
    "Tool",
    "ToolRegistry",
    "ToolResult",
]
