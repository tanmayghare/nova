from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from nova.core.browser import Browser
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """An autonomous agent that can interact with the web and use tools.
    
    This agent provides a core implementation for web automation and task execution,
    combining browser automation, LLM-based decision making, and tool usage.
    """

    def __init__(
        self,
        llm: LLM,
        tools: Optional[Sequence[Tool]] = None,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
    ) -> None:
        """Initialize the agent with LLM, tools, memory, and configuration.
        
        Args:
            llm: Language model for decision making
            tools: Collection of tools available to the agent
            memory: Memory system for context management
            config: Agent configuration
            browser_config: Browser configuration
        """
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or Memory()
        self.config = config or AgentConfig()
        self.browser = Browser(browser_config or BrowserConfig())
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            self.tool_registry.register(tool)

    async def start(self) -> None:
        """Start the agent and its browser."""
        await self.browser.start()

    async def stop(self) -> None:
        """Stop the agent and its browser."""
        await self.browser.stop()

    async def run(self, task: str) -> str:
        """Run the agent on a given task.
        
        Args:
            task: The task description to execute
            
        Returns:
            The result of executing the task
        """
        await self.start()
        try:
            result = await self._execute_task(task)
            return result
        finally:
            await self.stop()

    async def _execute_task(self, task: str) -> str:
        """Execute a task using the agent's capabilities.
        
        Args:
            task: The task description to execute
            
        Returns:
            The result of executing the task
        """
        # Get relevant context from memory
        context = await self.memory.get_context(task)
        
        # Generate a plan using the LLM
        plan = await self.llm.generate_plan(task, context)
        
        # Execute the plan step by step
        for step in plan:
            # Choose appropriate tool or action
            if step["type"] == "browser":
                result = await self._execute_browser_action(step["action"])
            elif step["type"] == "tool":
                result = await self._execute_tool(step["tool"], step["input"])
            else:
                result = "Unknown action type"
            
            # Store the result in memory
            await self.memory.add(task, step, result)
        
        # Generate final response
        final_response = await self.llm.generate_response(task, plan, context)
        return final_response

    async def _execute_browser_action(self, action: Dict[str, Any]) -> str:
        """Execute a browser action.
        
        Args:
            action: Dictionary containing the action type and parameters
            
        Returns:
            A description of the action's result
        """
        action_type = action.get("type")
        if action_type == "navigate":
            await self.browser.navigate(action["url"])
            return f"Navigated to {action['url']}"
        elif action_type == "click":
            await self.browser.click(action["selector"])
            return f"Clicked element {action['selector']}"
        elif action_type == "type":
            await self.browser.type(action["selector"], action["text"])
            return f"Typed '{action['text']}' into {action['selector']}"
        elif action_type == "get_text":
            text = await self.browser.get_text(action["selector"])
            return f"Got text: {text}"
        else:
            return f"Unknown browser action: {action_type}"

    async def _execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> str:
        """Execute a tool with the given input.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input parameters for the tool
            
        Returns:
            The tool's execution result
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return f"Tool not found: {tool_name}"
        return await tool.execute(input_data)

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.browser.close()
