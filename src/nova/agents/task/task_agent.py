"""Task Agent Implementation."""

from __future__ import annotations

import logging
import json
import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union

from ...core.llm import LLM, LLMConfig
from ...core.memory import Memory
from ...core.tools import Tool, ToolRegistry, ToolResult
from ...core.browser import Browser, BrowserConfig, BrowserTools
from ...core.exceptions import AgentError, LLMError, MaxStepsExceededError
from ..base.base_agent import BaseAgent, AgentState
from ...core.config.config import MemoryConfig

logger = logging.getLogger(__name__)

MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", 20)) # Max steps before halting

@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: str
    steps_taken: int
    result: Union[List[Any], str, None]
    error: Optional[str]
    history: List[Dict[str, Any]]
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Convert history messages to serializable format
        serialized_history = []
        for msg in self.history:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                serialized_history.append({
                    'type': msg.type,
                    'content': msg.content
                })
            else:
                serialized_history.append(str(msg))
                
        return {
            "task_id": self.task_id,
            "status": self.status,
            "steps_taken": self.steps_taken,
            "result": self.result,
            "error": self.error,
            "history": serialized_history
        }

class TaskAgent(BaseAgent):
    """Agent specialized for task execution using a ReAct loop.

    This agent:
    1. Takes a task description.
    2. Iteratively gathers context, generates a thought and the next action using LLM.
    3. Executes the action using available tools.
    4. Updates memory/history with the outcome.
    5. Repeats until the task is finished or an error occurs.
    """

    def __init__(
        self,
        task_id: str,
        task_description: str,
        llm_config: LLMConfig,
        browser_config: Optional[BrowserConfig] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """Initialize the task agent.
        
        Args:
            task_id: Unique identifier for this task
            task_description: Description of the task to execute
            llm_config: Configuration for the LLM
            browser_config: Optional configuration for browser automation
            memory: Optional memory instance
            tools: Optional list of additional tools
        """
        super().__init__(
            llm_config=llm_config,
            browser_config=browser_config,
            memory=memory,
            tools=tools or [], # Ensure tools is a list
        )
        
        self.task_id = task_id
        self.task_description = task_description
        self.max_steps = MAX_STEPS
        
        # Ensure BrowserTools are registered if browser_config is provided
        if self.browser_config and not self.tool_registry.get_tool_by_type(BrowserTools):
            logger.info("Browser config provided, initializing and registering BrowserTools.")
            self.browser = Browser(config=self.browser_config)
            browser_tools = BrowserTools(self.browser)
            self.tool_registry.register_tool(browser_tools)
        else:
            self.browser = None # No browser if no config

        # Ensure memory is initialized with proper config
        if not self.memory:
            memory_config = MemoryConfig()
            self.memory = Memory(config=memory_config)

        # State specific to TaskAgent loop
        self.action_history: List[Dict[str, Any]] = []
        self.current_step = 0

    async def start(self) -> None:
        """Start the agent and browser if configured."""
        await super().start()
        if self.browser:
            await self.browser.start()

    async def stop(self) -> None:
        """Stop the agent and browser."""
        if self.browser:
            await self.browser.stop()
        await super().stop()

    async def _gather_context(self, task: str) -> str:
        """Gather context for the LLM plan generation."""
        # 1. Get current page structure (simplified for now)
        page_structure = "No browser available."
        current_url = "N/A"
        dom_snapshot = ""
        if self.browser:
            try:
                current_url = await self.browser.get_current_url()
                # Get simplified DOM structure using a tool if available
                dom_tool = self.tool_registry.get_tool_by_type(BrowserTools)
                if dom_tool and hasattr(dom_tool, 'get_dom_snapshot'):
                     dom_snapshot = await dom_tool.get_dom_snapshot() 
                     # Limit size for prompt
                     dom_snapshot = (dom_snapshot[:2000] + '...') if len(dom_snapshot) > 2000 else dom_snapshot
                     page_structure = f'Current URL: {current_url}\nPage Structure Snippet:\n```html\n{dom_snapshot}\n```'
                else:
                    page_structure = f"Current URL: {current_url}\nPage Structure: Could not retrieve DOM snapshot."

            except Exception as e:
                logger.warning(f"Failed to get browser context: {e}")
                page_structure = f"Current URL: {current_url}\nPage Structure: Error retrieving - {e}"

        # 2. Format recent history (limit length)
        recent_history = self.action_history[-5:] # Get last 5 interactions
        history_str = json.dumps(recent_history, indent=2)

        # 3. Combine into a context string
        context = f"""Initial Task Context:
Task: {task}
Available Tools: {list(self.tool_registry.get_tool_names())}

Current Page Structure (After Previous Action):
{page_structure}

Recent Execution History (Thought/Action/Observation):
{history_str}
"""
        return context

    async def run(self) -> TaskResult:
        """Run the task."""
        try:
            await self.start()
            logger.info(f"Starting task '{self.task_id}': {self.task_description}")
            
            while self.current_step <= self.max_steps:
                logger.info(f"--- Step {self.current_step}/{self.max_steps} --- Task: {self.task_id}")
                
                # Get next action from LLM
                action = await self._get_next_action()
                if not action:
                    break
                    
                # Execute action
                try:
                    action_result = await self._execute_action(action)
                    logger.info(f"Step {self.current_step} - Action Result: Success={action_result['success']}, Output=\"{str(action_result['data'])[:100]}...\"")
                    
                    # Update memory with action result
                    self.memory.save_context(
                        {"input": str(action)},
                        {"output": str(action_result)}
                    )
                    
                    # Check if task is complete
                    if await self._is_task_complete(action_result):
                        break
                        
                except Exception as e:
                    logger.error(f"Unexpected error during action execution at step {self.current_step}: {str(e)}")
                    self._handle_error("unexpected_execution_error", {"step": self.current_step, "error": str(e)})
                    break
                    
                self.current_step += 1
                
            return self._get_task_result()
            
        except Exception as e:
            logger.error(f"Error running task: {str(e)}")
            return TaskResult(
                task_id=self.task_id,
                status="failed",
                steps_taken=self.current_step,
                result=[],
                error=str(e),
                history=self.memory.chat_memory.messages
            )

    async def _get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get the next action from the LLM."""
        try:
            context = await self._gather_context(self.task_description)
            thought, confidence, plan_steps = await self.llm.generate_plan(
                task=self.task_description,
                context=context
            )
            logger.info(f"Step {self.current_step} - Thought: {thought}")
            logger.info(f"Step {self.current_step} - Confidence: {confidence:.2f}")
            
            if not plan_steps:
                logger.warning("LLM did not return a plan step. Halting task.")
                return None
                
            next_step = plan_steps[0]  # Expecting only one step
            logger.info(f"Step {self.current_step} - Proposed Action: {next_step}")
            
            # Store thought and action intention
            self.action_history.append({
                "step": self.current_step,
                "thought": thought,
                "confidence": confidence,
                "action": next_step,
                "observation": None  # To be filled after execution
            })
            
            return next_step
            
        except Exception as e:
            logger.error(f"Failed to generate plan for step {self.current_step}: {e}")
            self._handle_error("plan_generation_failed", {"step": self.current_step, "error": str(e)})
            return None
            
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action using the tool registry."""
        tool_name = action.get("tool")
        tool_input = action.get("input", {})
        
        if tool_name == "finish":
            logger.info(f"Finish action received. Reason: {tool_input.get('reason')}")
            result = {
                "success": True,
                "data": {"reason": tool_input.get("reason", "Task marked as finished by agent.")},
                "error": ""
            }
            # Update the last action record with the result
            if self.action_history:
                self.action_history[-1]["observation"] = result
            return result
            
        try:
            result = await self.tool_registry.execute_tool(
                tool_name=tool_name,
                input_data=tool_input
            )
            # Update the last action record with the result
            if self.action_history:
                self.action_history[-1]["observation"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            if self.action_history:
                self.action_history[-1]["observation"] = {"status": "error", "error": str(e)}
            raise
            
    async def _is_task_complete(self, action_result: Dict[str, Any]) -> bool:
        """Check if the task is complete based on the action result."""
        # For now, only consider "finish" tool as completion
        return action_result.get("data", {}).get("reason") is not None
        
    def _get_task_result(self) -> TaskResult:
        """Get the final task result."""
        if not self.action_history:
            return TaskResult(
                task_id=self.task_id,
                status="failed",
                steps_taken=self.current_step,
                result=[],
                error="No actions were executed",
                history=self.memory.chat_memory.messages
            )
            
        last_action = self.action_history[-1]
        last_observation = last_action.get("observation", {})
        
        # Check if the last action was a finish action
        if last_action["action"]["tool"] == "finish":
            # Task completed with finish tool
            return TaskResult(
                task_id=self.task_id,
                status="completed",
                steps_taken=self.current_step,
                result=last_action["action"]["input"]["reason"],
                error=None,
                history=self.memory.chat_memory.messages
            )
        else:
            # Task ended without explicit finish
            return TaskResult(
                task_id=self.task_id,
                status="incomplete",
                steps_taken=self.current_step,
                result=self.action_history,
                error=None,
                history=self.memory.chat_memory.messages
            ) 