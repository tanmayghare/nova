"""Nova - An autonomous web agent framework."""

from .core.browser import Browser
from .core.config import AgentConfig
from .browser.config import BrowserConfig
from .core.llm.llm import LLM
from .core.memory import Memory
from .core.tools import Tool, ToolRegistry, ToolResult
from .agents.task.task_agent import TaskAgent
from .agents.base.base_agent import BaseAgent

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "TaskAgent",
    "Browser",
    "AgentConfig",
    "BrowserConfig",
    "LLM",
    "Memory",
    "Tool",
    "ToolRegistry",
    "ToolResult",
]
