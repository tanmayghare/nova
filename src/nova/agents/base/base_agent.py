"""Base Agent Implementation."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ...core.llm import LLM, LLMConfig
from ...core.memory import Memory
from ...core.tools import Tool, ToolRegistry
from ...core.browser import Browser, BrowserConfig

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Agent execution state."""
    task: Optional[str] = None
    status: str = "idle"
    result: Optional[Any] = None
    error: Optional[str] = None

class BaseAgent:
    """Base class for all agent implementations."""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        browser_config: Optional[BrowserConfig] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """Initialize the base agent."""
        self.llm = LLM(config=llm_config)
        self.browser_config = browser_config
        self.memory = memory
        self.tool_registry = ToolRegistry()
        self._state = AgentState()
        
        if tools:
            for tool in tools:
                self.tool_registry.register_tool(tool)
                
    async def start(self) -> None:
        """Start the agent."""
        logger.info("Agent started")
        
    async def stop(self) -> None:
        """Stop the agent."""
        logger.info("Agent stopped")
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop()
        
    def get_state(self) -> AgentState:
        """Get the current agent state."""
        return self._state
        
    async def run(self, task: str) -> Dict[str, Any]:
        """Run a task with the agent.
        
        This method should be implemented by subclasses.
        
        Args:
            task: The task description to execute
            
        Returns:
            Dictionary containing the execution results
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    def _log_action(self, action: str, details: Dict[str, Any]) -> None:
        """Log an agent action."""
        logger.info(f"Action: {action}, Details: {details}")
        
    def _log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error with context."""
        logger.error(f"Error: {error}, Context: {context}") 