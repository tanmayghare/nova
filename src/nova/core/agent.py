from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, Optional, Sequence, List
from datetime import datetime
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
        self._browser = None
        self.tool_registry = ToolRegistry()
        self.monitor = PerformanceMonitor()
        self.max_parallel_tasks = max_parallel_tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Register tools
        for tool in self.tools:
            self.tool_registry.register(tool)
            
        # Set browser if provided
        if browser:
            self.browser = browser

    @property
    def browser(self) -> Optional[Browser]:
        """Get the current browser instance."""
        return self._browser

    @browser.setter
    def browser(self, browser: Optional[Browser]) -> None:
        """Set the browser instance and register browser tools."""
        self._browser = browser
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
        if self.browser:
            await self.browser.start()
        else:
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
        """Execute a single task using a ReAct loop."""
        with self.monitor.track_task(task_id):
            max_iterations = self.config.max_iterations
            action_history = [] 
            cumulative_results = [] 
            current_context = await self.memory.get_context(task["description"]) # Initial context
            consecutive_tool_failures = 0 
            # Initialize dom_context_str for the first iteration
            dom_context_str = "DOM not queried yet (no action taken)."

            try:
                for i in range(max_iterations):
                    logger.info(f"ReAct Iteration {i+1}/{max_iterations} for task: {task_id}")
                    
                    # --- Prepare Context for THIS iteration's LLM call ---
                    # Get initial/memory context (fetched once initially)
                    initial_context = current_context # Use the initially fetched context
                                            
                    # Get Recent History
                    history_limit = self.config.max_history_context_iterations
                    if history_limit > 0 and i > 0: # Only include history after first iteration
                        recent_history = action_history[-history_limit:]
                        history_context_str = json.dumps(recent_history, indent=2)
                    else:
                        recent_history = [] 
                        history_context_str = "No history yet." if i == 0 else "History limit reached or disabled."
                        
                    # Combine context components (DOM string is from *previous* iteration)
                    context_for_llm = f"""
Initial Task Context:
{initial_context}

Current Page Structure (After Previous Action):
```json
{dom_context_str}
```

Recent Execution History (last {len(recent_history)} steps):
```json
{history_context_str}
```
"""
                    logger.debug(f"Task {task_id} - Context for LLM (Iteration {i+1}):\n{context_for_llm}")
                    # --- End Context Preparation ---

                    # Step 1: Thought & Action Generation (LLM Call)
                    try:
                        thought, plan_steps = await self.llm.generate_plan(task["description"], context_for_llm)
                        logger.info(f"Task {task_id} - Thought: {thought}")
                        # Store thought immediately, action/observation added later
                        current_history_entry = {"iteration": i+1, "thought": thought}
                        action_history.append(current_history_entry)

                        if not plan_steps:
                            logger.warning(f"Task {task_id} - LLM did not provide a next step. Ending task.")
                            break 

                        # For ReAct, we expect one action per iteration
                        next_step = plan_steps[0]
                        if len(plan_steps) > 1:
                             logger.warning(f"Task {task_id} - LLM returned multiple steps ({len(plan_steps)}). Will execute only the first.")
                        
                        # --- Add finish tool check --- 
                        if next_step.get("tool") == "finish":
                            logger.info(f"Task {task_id} - LLM signaled completion. Reason: {next_step.get('input', {}).get('reason', 'None provided')}")
                            current_history_entry["action"] = next_step # Record the finish action
                            current_history_entry["observation"] = {"status": "success", "result": "Task marked as finished by LLM."}
                            break # Exit the loop successfully
                        # --- End finish tool check ---
                        
                        current_history_entry["action"] = next_step # Add action to current history entry

                    except Exception as e:
                        logger.error(f"Task {task_id} - Failed to generate thought/action: {e}", exc_info=True)
                        # Ensure history entry exists before adding error
                        if not action_history or action_history[-1]["iteration"] != i+1:
                             action_history.append({"iteration": i+1, "error": f"LLM generation failed: {e}"})
                        else:
                             action_history[-1]["error"] = f"LLM generation failed: {e}"
                        # Stop the task on LLM failure
                        return {"status": "failed", "error": f"LLM generation failed: {e}", "history": action_history}

                    # Step 2: Execute Action (only if not 'finish')
                    tool_name = next_step.get("tool")
                    tool_input = next_step.get("input", {})
                    
                    if not tool_name:
                         logger.error(f"Task {task_id} - Invalid step format from LLM (missing 'tool'): {next_step}")
                         action_history[-1]["observation"] = {"error": "Invalid action format from LLM (missing 'tool')"}
                         consecutive_tool_failures += 1 # Count this as a failure type
                         # Check failure limit immediately after incrementing
                         if consecutive_tool_failures >= self.config.max_failures:
                             logger.error(f"Task {task_id} - Reached max tool failures ({self.config.max_failures}) after invalid action format. Stopping.")
                             return {"status": "failed", "error": f"Reached max tool failures ({self.config.max_failures}) after invalid action format.", "history": action_history}
                         continue 

                    try:
                        logger.info(f"Task {task_id} - Executing Action: {tool_name}({tool_input})")
                        result: ToolResult = await self.tool_registry.execute_tool(tool_name, tool_input)
                        
                        # --- Action SUCCEEDED --- 
                        consecutive_tool_failures = 0 # Reset counter
                        
                        # --- Get DOM and Screenshot AFTER successful action ---
                        # Get Structured DOM for the *next* iteration's context
                        try:
                            logger.debug("Getting DOM structure after successful action...")
                            dom_structure = await self._get_structured_dom()
                            dom_context_str = json.dumps(dom_structure, indent=2)
                            logger.debug("DOM structure updated for next context.")
                        except Exception as dom_e:
                            logger.error(f"Task {task_id} - Failed to get structured DOM after action: {dom_e}", exc_info=True)
                            dom_context_str = "Error fetching DOM structure after action."

                        # Screenshot Logic (V1 - store path, don't store bytes yet)
                        screenshot_path_for_history = "Not taken or failed"
                        if self.config.use_vision and self.browser:
                            try:
                                screenshot_filename = f"screenshot_{task_id}_iter_{i+1}.png"
                                # Call screenshot, but ignore the returned bytes for now
                                await self.browser.screenshot(path=screenshot_filename) 
                                screenshot_path_for_history = screenshot_filename # Store the intended path
                                logger.info(f"Task {task_id} - Screenshot saved to: {screenshot_path_for_history}")
                            except Exception as ss_e:
                                logger.warning(f"Task {task_id} - Failed to take screenshot: {ss_e}")
                        
                        # --- Prepare Observation (Make result data JSON serializable) ---
                        serializable_data = None
                        if isinstance(result.data, (str, int, float, bool, list, dict, type(None))):
                            serializable_data = result.data
                        elif isinstance(result.data, bytes):
                            serializable_data = f"<bytes data len={len(result.data)}>"
                        else:
                            # Convert other types to string representation
                            try:
                                serializable_data = str(result.data)
                            except Exception:
                                serializable_data = f"<unserializable data type: {type(result.data).__name__}>"
                        
                        observation = {
                            "status": "success", 
                            "result": serializable_data, # Use the sanitized data
                            "error": result.error, # Include error field from ToolResult if any
                            "screenshot": screenshot_path_for_history 
                        }
                        await self.memory.add(task_id, next_step, observation) 
                        action_history[-1]["observation"] = observation
                        cumulative_results.append(serializable_data) # Store serializable result
                        logger.info(f"Task {task_id} - Action Result Data: {serializable_data}")
                        
                    except Exception as e:
                        # --- Action FAILED --- 
                        # Handle tool execution failure
                        logger.error(f"Task {task_id} - Step execution failed: {tool_name}({tool_input}) - {e}", exc_info=True)
                        observation = {"status": "error", "error": str(e)}
                        await self.memory.add(task_id, next_step, observation)
                        action_history[-1]["observation"] = observation
                        
                        # Increment failure counter and check limit
                        consecutive_tool_failures += 1
                        logger.warning(f"Task {task_id} - Consecutive tool failures: {consecutive_tool_failures}/{self.config.max_failures}")
                        if consecutive_tool_failures >= self.config.max_failures:
                            logger.error(f"Task {task_id} - Reached max tool failures ({self.config.max_failures}). Stopping task.")
                            return {"status": "failed", "error": f"Reached max tool failures ({self.config.max_failures}) after error: {e}", "history": action_history}
                        
                        # Keep the previous dom_context_str for the next iteration's context
                        logger.warning("Keeping previous DOM context after tool failure.")
                        
                    # Step 3 is now integrated: DOM/Screenshot taken after success, context built at start of loop

                # After loop (max iterations reached or break condition)
                logger.info(f"Task {task_id} - ReAct loop finished after {len(action_history)} iterations.")

                # Step 4: Generate Final Response
                # Use the accumulated history as the 'plan'/'context' for the final response
                final_response_context = f"Task completed or max iterations reached. History:\\n{json.dumps(action_history, indent=2)}"
                response = await self.llm.generate_response(task["description"], action_history, final_response_context) # Pass history as 'plan'
                
                return {
                    "status": "success", # Or determine status based on last observation?
                    "results": cumulative_results,
                    "response": response,
                    "history": action_history
                }
                
            except Exception as e:
                logger.error(f"Task {task_id} - Unhandled exception during execution: {e}", exc_info=True)
                return {
                    "status": "failed",
                    "error": str(e),
                    "history": action_history # Include history even on failure
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
            logger.error("Browser action failed", exc_info=True)
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
            logger.error("Tool action failed", exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.browser:
            await self.browser.stop()
        else:
            await self.browser_pool.stop()
        self.monitor.stop()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.monitor.get_metrics()

    # --- Add Method to Extract Structured DOM --- 
    async def _get_structured_dom(self, max_elements: int = 100) -> List[Dict[str, Any]]:
        """Extract structured information about interactive elements from the current page."""
        if not self.browser or not self.browser.page:
            logger.warning("Cannot get structured DOM: Browser or page not available.")
            return []

        page = self.browser.page
        structured_elements = []
        kept_attributes = ['id', 'name', 'class', 'type', 'placeholder', 'aria-label', 'role', 'href', 'alt', 'title']

        try:
            # Query for potentially interactive elements
            # Expand selectors as needed
            selectors = 'a, button, input, textarea, select, [role="button"], [role="link"], [role="textbox"], [role="menuitem"]'
            element_handles = await page.query_selector_all(selectors)
            logger.info(f"Found {len(element_handles)} potential interactive elements.")

            count = 0
            for handle in element_handles:
                if count >= max_elements:
                    logger.warning(f"Reached max elements ({max_elements}) for structured DOM.")
                    break
                try:
                    is_visible = await handle.is_visible()
                    is_enabled = await handle.is_enabled()
                    
                    # Filter out non-visible or disabled elements (common criteria)
                    if not is_visible or not is_enabled:
                        continue

                    # Get basic info
                    tag_name = (await handle.evaluate('element => element.tagName')).lower()
                    attributes = await handle.evaluate(f'element => {{ const attrs = {{}}; Array.from(element.attributes).forEach(attr => {{ if ({json.dumps(kept_attributes)}.includes(attr.name)) attrs[attr.name] = attr.value; }}); return attrs; }}')
                    text_content = (await handle.text_content() or "").strip()
                    bbox = await handle.bounding_box() # Format: {'x', 'y', 'width', 'height'}
                    
                    # Add placeholder to text content if it's empty and placeholder exists
                    if not text_content and attributes.get('placeholder'):
                        text_content = f"(placeholder: {attributes['placeholder']})"
                        
                    # Basic filtering based on bounding box (ignore tiny elements if needed)
                    if not bbox or bbox['width'] < 2 or bbox['height'] < 2:
                         continue

                    element_data = {
                        "tag": tag_name,
                        "attributes": attributes,
                        "text": text_content,
                        "bbox": bbox
                    }
                    structured_elements.append(element_data)
                    count += 1

                except Exception as e:
                    # Log error for specific element but continue with others
                    logger.warning(f"Error processing element for structured DOM: {e}", exc_info=False) # Avoid excessive logging
                finally:
                    # Ensure handles are disposed if playwright-python requires it (check documentation)
                    # await handle.dispose() # Uncomment if necessary
                    pass 

            logger.info(f"Extracted {len(structured_elements)} elements for structured DOM.")
            return structured_elements

        except Exception as e:
            logger.error(f"Failed to query or process elements for structured DOM: {e}", exc_info=True)
            return []
    # --- End Method --- 
