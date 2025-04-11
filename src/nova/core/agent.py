from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, Optional, Sequence, List
from datetime import datetime, timedelta
import uuid
import json

from .browser import Browser, BrowserPool
from .config import AgentConfig, BrowserConfig
from .llm import LLM
from .memory import Memory
from .tools import Tool, ToolRegistry
from .monitoring import PerformanceMonitor
from ..tools.browser import NavigateTool, ClickTool, TypeTool, WaitTool, ScreenshotTool

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
        browser: Optional[Browser] = None,
        max_parallel_tasks: int = 3,
        browser_pool_size: int = 5,
    ) -> None:
        """Initialize the agent with LLM, tools, memory, and configuration.
        
        Args:
            llm: Language model for decision making
            tools: Collection of tools available to the agent
            memory: Memory system for context management
            config: Agent configuration
            browser_config: Browser configuration
            browser: Optional pre-configured browser instance
            max_parallel_tasks: Maximum number of tasks to execute in parallel
            browser_pool_size: Size of the browser connection pool
        """
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or Memory()
        self.config = config or AgentConfig()
        self.browser_pool = BrowserPool(
            size=browser_pool_size,
            config=browser_config or BrowserConfig()
        )
        self.browser = browser
        self.tool_registry = ToolRegistry()
        self.monitor = PerformanceMonitor()
        self.max_parallel_tasks = max_parallel_tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Register tools
        for tool in self.tools:
            self.tool_registry.register(tool)
            
        # Register browser tools
        if browser:
            self._register_browser_tools(browser)

    def _register_browser_tools(self, browser: Browser) -> None:
        """Register browser action tools."""
        browser_tools = [
            NavigateTool(browser),
            ClickTool(browser),
            TypeTool(browser),
            WaitTool(browser),
            ScreenshotTool(browser)
        ]
        for tool in browser_tools:
            self.tool_registry.register(tool)
            logger.info(f"Registered browser tool: {tool.config.name}")

    async def start(self) -> None:
        """Start the agent and its browser pool."""
        if self.browser_pool.started:
            logger.warning("Browser pool already started")
            return
        await self.browser_pool.start()
        self.monitor.start()

    async def stop(self) -> None:
        """Stop the agent and its browser pool."""
        await self.browser_pool.stop()
        self.monitor.stop()
        
        # Cancel any active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.active_tasks.clear()

    async def run(self, task: str, task_id: Optional[str] = None) -> str:
        """Run the agent on a given task.
        
        Args:
            task: The task description to execute
            task_id: Optional task identifier for tracking
            
        Returns:
            The result of executing the task
            
        Raises:
            Exception: If task execution fails
        """
        task_id = task_id or str(uuid.uuid4())
        
        if task_id in self.active_tasks:
            raise RuntimeError(f"Task {task_id} is already running")
            
        try:
            await self.start()
            
            # Convert string task to dictionary with description
            task_dict = {
                "description": task,
                "id": task_id,
                "created_at": datetime.now().isoformat()
            }
            
            # Create and track the task
            task_obj = asyncio.create_task(self._execute_task(task_dict, task_id))
            self.active_tasks[task_id] = task_obj
            
            # Wait for completion
            result = await task_obj
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            raise
        finally:
            self.active_tasks.pop(task_id, None)
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}", exc_info=True)

    async def run_batch(self, tasks: List[str]) -> Dict[str, str]:
        """Run multiple tasks in parallel.
        
        Args:
            tasks: List of task descriptions to execute
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}
            
        # Limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        
        async def run_with_semaphore(task: str) -> tuple[str, str]:
            async with semaphore:
                task_id = str(uuid.uuid4())
                result = await self.run(task, task_id)
                return task_id, result
                
        # Run tasks in parallel
        task_results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        results = {}
        for task_result in task_results:
            if isinstance(task_result, Exception):
                logger.error(f"Task failed: {task_result}", exc_info=True)
                continue
            task_id, result = task_result
            results[task_id] = result
            
        return results

    async def _execute_task(self, task: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute a single task."""
        with self.monitor.track_task(task_id):
            try:
                # Get context from memory
                context = await self.memory.get_context(task["description"])
                
                # Generate plan using LLM
                plan = await self.llm.generate_plan(task, context)
                
                # Execute each step in the plan
                results = []
                for step in plan:
                    try:
                        # Ensure step is a dictionary
                        if isinstance(step, str):
                            try:
                                step = json.loads(step)
                            except json.JSONDecodeError:
                                raise ValueError(f"Invalid step format: {step}")
                        
                        if not isinstance(step, dict):
                            raise ValueError(f"Invalid step format: {type(step)}")
                        
                        if "type" not in step or step["type"] != "tool":
                            raise ValueError(f"Invalid step type: {step.get('type', 'missing')}")
                        
                        # Execute tool
                        result = await self.tool_registry.execute_tool(
                            step["tool"],
                            step["input"]
                        )
                        
                        # Store result in memory
                        await self.memory.add(task_id, step, result)
                        results.append(result)
                        
                    except Exception as e:
                        error_msg = f"Step execution failed: {str(e)}"
                        logger.error(error_msg)
                        await self.memory.add(task_id, step, {"error": error_msg})
                        raise
                
                return {
                    "status": "success",
                    "results": results
                }
                
            except Exception as e:
                error_msg = f"Task execution failed: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "failed",
                    "error": error_msg
                }

    async def _execute_browser_action(self, browser: Browser, action: Dict[str, Any]) -> Any:
        """Execute a browser action."""
        try:
            action_type = action.get("type")
            if action_type == "navigate":
                return await browser.navigate(action["url"])
            elif action_type == "click":
                return await browser.click(action["selector"])
            elif action_type == "type":
                return await browser.type(action["selector"], action["text"])
            elif action_type == "wait":
                return await browser.wait(action["selector"], action.get("timeout", 10))
            elif action_type == "screenshot":
                return await browser.screenshot(action.get("path"))
            else:
                raise ValueError(f"Unknown browser action type: {action_type}")
        except Exception as e:
            logger.error(f"Browser action failed: {e}", exc_info=True)
            raise

    async def _execute_tool_action(self, action: Dict[str, Any]) -> Any:
        """Execute a tool action.
        
        Args:
            action: Dictionary containing the tool action details
            
        Returns:
            Result of the tool action
            
        Raises:
            ValueError: If the tool or action is not found
            Exception: If the tool action fails
        """
        try:
            tool_name = action.get("tool")
            if not tool_name:
                raise ValueError("Tool name not specified in action")
                
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
                
            # Execute the tool action
            return await tool.execute(action)
            
        except Exception as e:
            logger.error(f"Tool action failed: {e}", exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.browser_pool:
                await self.browser_pool.cleanup()
        except Exception as e:
            logger.error(f"Browser pool cleanup failed: {e}", exc_info=True)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.monitor.get_metrics()
