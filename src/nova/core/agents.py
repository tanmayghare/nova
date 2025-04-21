import asyncio
import logging
import uuid
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph

from .llm import LLM, LLMConfig
from .memory import Memory
from .tools import ToolRegistry, ToolConfig, ToolResult
from .browser import Browser
from nova.tools.browser import get_browser_tools
from nova.tools.utils.web_search import WebSearchTool

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """State management for agent execution."""
    task_id: str
    task_description: str
    current_action: Optional[Dict[str, Any]] = None
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    consecutive_failures: int = 0
    final_outcome: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dom_content: Optional[str] = None
    latest_screenshot_bytes: Optional[bytes] = None
    browser: Optional[Browser] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    parallel_tasks: List[asyncio.Task] = field(default_factory=list)
    context: str = "" # Added context field to store history
    result: Optional[Dict[str, Any]] = None # Added result field for LangGraph state passing
    
    def add_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Add an action and its result to the history."""
        self.current_action = action
        self.action_history.append({
            "action": action,
            "result": result,
            "timestamp": str(uuid.uuid4())
        })
        
        # Check if the result indicates an error
        # Use .get() for safer access
        is_error = False
        if isinstance(result, dict):
            # Handle both simple error strings and structured ToolResult errors
            if result.get("error"):
                is_error = True 
            elif result.get("status") == "error":
                 is_error = True

        if is_error:
            self.consecutive_failures += 1
            # Store the error message, preferring a dedicated 'error' key
            self.error = result.get("error") if isinstance(result, dict) else "Unknown error structure"
            if not self.error and isinstance(result, dict):
                 # Fallback if error is nested in data
                 self.error = result.get("data", {}).get("error")
            if not self.error: # Ultimate fallback
                 self.error = json.dumps(result) 
        else:
            self.consecutive_failures = 0
            self.error = None
            
    def set_final_outcome(self, outcome: Dict[str, Any]) -> None:
        """Set the final outcome of the task."""
        self.final_outcome = outcome
        
    def get_last_action(self) -> Optional[Dict[str, Any]]:
        """Get the last action from history."""
        return self.action_history[-1] if self.action_history else None
        
    def add_parallel_task(self, task: asyncio.Task) -> None:
        """Add a parallel task to track."""
        self.parallel_tasks.append(task)
        
    async def cleanup_parallel_tasks(self) -> None:
        """Clean up all parallel tasks."""
        for task in self.parallel_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.parallel_tasks.clear()

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str
    llm_config: LLMConfig
    max_iterations: int = 10
    max_retries: int = 3
    tools: Optional[List[ToolConfig]] = None

class BaseAgent:
    """Base agent class using LangGraph for orchestration."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        self.config = config
        self.browser = browser
        self.llm = LLM(config.llm_config)
        self.memory = Memory(
            llm=self.llm.get_langchain_llm(),
            memory_key="history"
        )
        self.tool_registry = ToolRegistry()
        
        # --- Register Tools --- 
        # Register BrowserTools instance (this handles registering its sub-tools)
        try:
            browser_tools_instance = get_browser_tools(self.browser)
            self.tool_registry.register_tool(browser_tools_instance)
        except Exception as e:
             logger.error(f"Failed to register BrowserTools: {e}", exc_info=True)
             # Decide if we should raise error or continue without browser tools

        # Register WebSearchTool instance
        try:
            web_search_tool_instance = WebSearchTool()
            self.tool_registry.register_tool(web_search_tool_instance)
        except Exception as e:
             logger.error(f"Failed to register WebSearchTool: {e}", exc_info=True)
             # Decide if we should raise error or continue without web search

        # Register any other tools passed via AgentConfig
        # These should likely be tool *instances* or have a standard way to be instantiated.
        # Current AgentConfig passes ToolConfig, which ToolRegistry.register_tool doesn't directly handle
        # for top-level tools. This part needs rethinking if external tools are passed via config.
        # For now, comment out or adjust based on how external tools are provided.
        # if config.tools:
        #     for tool_config in config.tools:
        #         # Avoid double-registering 
        #         if tool_config.name not in self.tool_registry.get_tool_names(): 
        #             # How to get/create the Tool instance from ToolConfig here?
        #             # This needs a factory or assumes config provides instances.
        #             logger.warning(f"Registration of external tools via config needs implementation.")
        #             # self.tool_registry.register_tool(???) 
                
        self._init_graph()
        
    # --- Routing Functions for Conditional Edges ---
    def _route_after_execution(self, state: AgentState) -> str:
        """Determines the next step after action execution based on errors."""
        # Access the result merged into the state by LangGraph
        action_result = state.result if state.result is not None else {}
        
        # Check for errors within the result dictionary
        is_error = False
        if isinstance(action_result, dict):
            if action_result.get("error") or action_result.get("status") == "error":
                is_error = True
                # Ensure the error is also set in the main state.error field for routing
                state.error = action_result.get("error", "Unknown error in tool result")
            else:
                state.error = None # Clear previous error if action succeeded
        else:
             # Handle non-dict results as potential errors?
             logger.warning(f"[{state.task_id}] Unexpected result type in _route_after_execution: {type(action_result)}. Assuming error.")
             is_error = True
             state.error = f"Unexpected result type: {type(action_result)}"

        if is_error:
            logger.warning(f"[{state.task_id}] Action failed (Error: {state.error}), routing to handle_error.")
            return "handle_error"
        else:
            logger.info(f"[{state.task_id}] Action successful, routing to update_state.")
            return "update_state"

    def _route_after_error_handling(self, state: AgentState) -> str:
        """Determines the next step after attempting error recovery."""
        # _can_recover checks if attempts < max and error exists and not finalized
        if self._can_recover(state):
            logger.info(f"[{state.task_id}] Recovery possible, routing back to generate_action.")
            return "generate_action" # Try generating a new action (could be recovery plan)
        else:
            logger.error(f"[{state.task_id}] Recovery not possible or max attempts reached, routing to cleanup.")
            return "cleanup"
            
    # --- Graph Definition ---
    def _init_graph(self) -> None:
        """Initialize the LangGraph state machine with enhanced error handling."""
        self.graph = StateGraph(AgentState)
        
        # Add nodes
        self.graph.add_node("prepare_context", self._prepare_context)
        self.graph.add_node("generate_action", self._generate_action)
        self.graph.add_node("execute_action", self._execute_action)
        self.graph.add_node("update_state", self._update_state)
        self.graph.add_node("check_termination_node", self._check_termination_node)
        self.graph.add_node("handle_error", self._handle_error)
        self.graph.add_node("cleanup", self._cleanup)
        
        # Define Edges
        self.graph.add_edge("prepare_context", "generate_action")
        self.graph.add_edge("generate_action", "execute_action")
        
        # Conditional edge after execution: success -> update_state, error -> handle_error
        self.graph.add_conditional_edges(
            "execute_action",
            self._route_after_execution,
            {
                "update_state": "update_state",
                "handle_error": "handle_error",
            }
        )
        
        # Edge after successful update, leading to the termination check node
        self.graph.add_edge("update_state", "check_termination_node")
        
        # Conditional edge after error handling: recovery possible -> generate_action, else -> cleanup
        self.graph.add_conditional_edges(
            "handle_error",
            self._route_after_error_handling,
            {
                "generate_action": "generate_action",
                "cleanup": "cleanup",
            }
        )
        
        # Conditional edge AFTER termination check node, using the new routing function
        self.graph.add_conditional_edges(
            "check_termination_node", # Source node is the new node
            self._route_after_termination_check, # Use the new routing function
            {
                "continue": "prepare_context", 
                "finish": "cleanup"
            }
        )
        
        self.graph.set_entry_point("prepare_context")
        self.graph.set_finish_point("cleanup") 
        
        # Compile the graph after defining structure
        self.graph = self.graph.compile()
        
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors and attempt recovery."""
        # Log the error from the state
        error_msg = state.error if state.error else "Unknown error during execution."
        logger.error(f"[{state.task_id}] Handling error: {error_msg}")
        
        # Increment consecutive failures (moved from add_action to ensure it happens on error path)
        state.consecutive_failures += 1
        logger.warning(f"[{state.task_id}] Consecutive failures incremented to: {state.consecutive_failures}")

        if state.recovery_attempts < state.max_recovery_attempts:
            state.recovery_attempts += 1
            logger.info(f"[{state.task_id}] Attempting recovery {state.recovery_attempts}/{state.max_recovery_attempts}...")
            # Clear the error for the retry attempt? Or let LLM see it?
            # Let LLM see the error to potentially generate a better recovery plan.
            # state.error = None # Optional: Clear error before retry?
            # Maybe generate a recovery plan here using LLM if desired
            # For now, just increment attempts and route back via _route_after_error_handling
        else:
            # If max recovery attempts reached, mark for cleanup
            logger.error(f"[{state.task_id}] Max recovery attempts reached. Marking for cleanup.")
            state.set_final_outcome({
                "status": "error",
                "message": f"Task failed after {state.consecutive_failures} consecutive errors and {state.recovery_attempts} recovery attempts.",
                "error": state.error
            })
            
        # Return the updated state (even if just recovery_attempts changed)
        # The actual routing happens in _route_after_error_handling
        return state 
        
    async def _cleanup(self, state: AgentState) -> AgentState:
        """Clean up resources and finalize state."""
        logger.info(f"[{state.task_id}] Starting cleanup...")
        try:
            # Clean up parallel tasks
            await state.cleanup_parallel_tasks()
            logger.info(f"[{state.task_id}] Parallel tasks cleaned up.")
            
            # Release browser if needed
            if state.browser:
                logger.info(f"[{state.task_id}] Stopping browser...")
                await state.browser.stop() # Use stop() instead of close()
                logger.info(f"[{state.task_id}] Browser stopped.")
                
        except Exception as e:
            logger.error(f"Error during resource cleanup (tasks/browser): {e}")
            
        # Update memory with final state - using save_context (synchronous)
        if state.final_outcome:
            final_state_input = {"input": f"Task finished with status: {state.final_outcome.get('status', 'unknown')}"}
            final_state_output = {"output": json.dumps(state.final_outcome)}
            try:
                self.memory.save_context(final_state_input, final_state_output) # Removed await
                logger.info(f"[{state.task_id}] Saved final state to memory.")
            except Exception as mem_e:
                logger.error(f"[{state.task_id}] Error saving final state to memory during cleanup: {mem_e}")
        else:
             # Ensure final_outcome is set if task failed unexpectedly before cleanup was called
             if not state.final_outcome and state.error:
                  state.set_final_outcome({
                     "status": "error",
                     "message": f"Task ended unexpectedly with error: {state.error}",
                     "error": state.error
                  })
                  logger.warning(f"[{state.task_id}] Final outcome was missing, set based on existing error before saving to memory.")
                  # Retry saving to memory now that final_outcome is set
                  final_state_input = {"input": f"Task finished with status: {state.final_outcome.get('status', 'unknown')}"}
                  final_state_output = {"output": json.dumps(state.final_outcome)}
                  try:
                       self.memory.save_context(final_state_input, final_state_output)
                       logger.info(f"[{state.task_id}] Saved final state to memory after setting fallback outcome.")
                  except Exception as mem_e:
                       logger.error(f"[{state.task_id}] Error saving final state to memory during cleanup (after fallback): {mem_e}")
             else:
                  logger.warning(f"[{state.task_id}] No final_outcome recorded and no state.error, cannot save final state to memory.")

        logger.info(f"[{state.task_id}] Cleanup finished.")
        return state # Return the final state after cleanup
        
    def _has_error(self, state: AgentState) -> bool:
        """Check if the state has an error."""
        # Check both state.error and if state.result indicates an error
        has_state_error = state.error is not None
        has_result_error = False
        if isinstance(state.result, dict):
             if state.result.get("error") or state.result.get("status") == "error":
                  has_result_error = True
        return has_state_error or has_result_error
        
    def _can_recover(self, state: AgentState) -> bool:
        """Check if the error can be recovered from."""
        # Check attempts, ensure error exists, and ensure task hasn't already finished
        return (
            state.recovery_attempts < state.max_recovery_attempts and
            self._has_error(state) and # Use helper to check error status
            state.final_outcome is None
        )
        
    def _cannot_recover(self, state: AgentState) -> bool:
        """Check if the error cannot be recovered from."""
        return not self._can_recover(state)
        
    async def run(self, task_id: str, task: str) -> Dict[str, Any]:
        """Run the agent on a task with enhanced error handling."""
        
        # Initialize state using the provided task_id
        state = AgentState(
            task_id=task_id, # Use the passed task_id
            task_description=task,
            browser=self.browser
        )
        
        logger.info(f"Starting agent run for task ID {state.task_id}: {task}")
        
        final_outcome_dict = {} # Use a different name to avoid confusion with state field
        try:
            # Ensure the browser is started before invoking the graph
            if not self.browser._browser: # Check if Playwright browser is actually running
                 logger.info(f"[{state.task_id}] Starting browser for agent run...")
                 await self.browser.start()
                 logger.info(f"[{state.task_id}] Browser started.")
            else:
                 logger.info(f"[{state.task_id}] Browser already started.")

            # The initial state passed to ainvoke should match the AgentState definition
            final_state: AgentState = await self.graph.ainvoke(state)
            
            # Ensure final_state is indeed AgentState and has final_outcome
            # The graph should ideally always return the final state object
            if final_state and hasattr(final_state, 'final_outcome') and final_state.final_outcome:
                if final_state.final_outcome.get("status") != "error":
                    final_outcome_dict = {
                        "status": "success",
                        "result": final_state.final_outcome.get("message", "Task completed."),
                        "action_history": final_state.action_history
                    }
                    logger.info(f"Agent run successful for task ID {state.task_id}. Outcome: {final_outcome_dict.get('result')}")
                else:
                    error_message = final_state.final_outcome.get("message", "Agent failed unexpectedly.")
                    # Prefer error from final_outcome if available
                    if final_state.final_outcome.get("error"):
                        error_message = final_state.final_outcome.get("error")
                    elif final_state.error: # Use the error field from AgentState as fallback
                        error_message = final_state.error
                        
                    final_outcome_dict = {
                        "status": "error",
                        "error": error_message,
                        "action_history": final_state.action_history
                    }
                    logger.error(f"Agent run failed for task ID {state.task_id}. Error: {error_message}")
            else:
                 # Handle cases where the graph might return None or unexpected structure
                 # Or if final_outcome wasn't set correctly during cleanup
                 logger.error(f"Agent run for task ID {state.task_id} finished with unexpected final state structure or missing outcome: {final_state}")
                 final_outcome_dict = {
                    "status": "error",
                    "error": state.error if state.error else "Agent finished with an unexpected internal state or missing outcome.",
                    "action_history": state.action_history # Use initial state history as fallback
                }
                
        except Exception as e:
            logger.error(f"Agent execution failed critically for task ID {state.task_id}: {e}", exc_info=True)
            # Use the initial state object which might have partial history
            final_outcome_dict = {
                "status": "error",
                "error": str(e),
                "action_history": state.action_history # Use initial state history
            }
            # Ensure final_outcome is set in the state object for cleanup
            if not state.final_outcome:
                 state.set_final_outcome({
                     "status": "error",
                     "message": f"Critical agent failure: {str(e)}",
                     "error": str(e)
                 })
            
        finally:
            # Ensure cleanup happens even if there's an error
            # Pass the state object that was potentially modified during the run/exception handling
            await self._cleanup(state) 
            
        return final_outcome_dict # Return the dictionary created

    async def _prepare_context(self, state: AgentState) -> Dict[str, Any]:
        """Prepare context for action generation, including DOM and screenshot."""
        logger.info(f"[{state.task_id}] Preparing context...")
        # Use load_memory_variables (synchronous) to get history
        try:
             memory_vars = self.memory.load_memory_variables({}) # Removed await
             context = memory_vars.get(self.memory.memory_key, "") 
             logger.debug(f"[{state.task_id}] Retrieved context from memory: {context[:200]}...")
             state.context = context # Store context directly in state
        except Exception as e:
             logger.error(f"[{state.task_id}] Failed to load context from memory: {e}", exc_info=True)
             state.context = "" # Default to empty context on error
        
        # Get BrowserTools instance from the registry dictionary
        try:
            # Assuming BrowserTools instance is registered under this exact name
            browser_tools_instance = self.tool_registry.tools["browser_tools"]
        except KeyError:
            logger.error(f"[{state.task_id}] Critical Error: BrowserTools not found in ToolRegistry.")
            # Handle this critical failure - maybe raise exception or set error state?
            state.error = "BrowserTools are essential but not registered."
            state.final_outcome = {"status": "error", "message": state.error}
            # Returning the error state might trigger the error handling path in the graph
            return {"error": state.error, "final_outcome": state.final_outcome}
        except Exception as e:
            logger.error(f"[{state.task_id}] Error retrieving BrowserTools from registry: {e}", exc_info=True)
            state.error = f"Failed to retrieve BrowserTools: {e}"
            state.final_outcome = {"status": "error", "message": state.error}
            return {"error": state.error, "final_outcome": state.final_outcome}

        # Get current DOM state using the retrieved browser tool instance
        try:
            # Ensure the tool instance has the method before calling
            if not hasattr(browser_tools_instance, 'get_dom_snapshot'):
                 raise AttributeError("BrowserTools instance does not have 'get_dom_snapshot' method")
            dom_result: ToolResult = await browser_tools_instance.get_dom_snapshot()
            if dom_result.success:
                state.dom_content = dom_result.data.get("dom_snapshot")
                logger.info(f"[{state.task_id}] DOM snapshot retrieved (length: {len(state.dom_content) if state.dom_content else 0}).")
            else:
                 logger.warning(f"[{state.task_id}] Failed to retrieve DOM snapshot: {dom_result.error}")
                 state.dom_content = None
                 # Optionally set state.error if DOM failure is critical
                 # state.error = f"DOM snapshot failed: {dom_result.error}"
        except AttributeError as ae:
            logger.error(f"[{state.task_id}] Error calling get_dom_snapshot: {ae}", exc_info=True)
            state.dom_content = None
            state.error = str(ae) # Set state error if tool method missing
        except Exception as e:
            logger.error(f"[{state.task_id}] Error getting DOM snapshot in prepare_context: {e}", exc_info=True)
            state.dom_content = None
            # Optionally set state.error
            # state.error = f"Error getting DOM snapshot: {e}"
            
        # Get current Screenshot (only if API service is selected, to avoid overhead)
        state.latest_screenshot_bytes = None # Reset first
        if self.llm.config.service_type == "api":
            logger.info(f"[{state.task_id}] Attempting to get screenshot for API LLM...")
            try:
                # Ensure the tool instance has the method before calling
                if not hasattr(browser_tools_instance, 'screenshot'):
                     raise AttributeError("BrowserTools instance does not have 'screenshot' method")
                screenshot_result: ToolResult = await browser_tools_instance.screenshot(path=None) # Get bytes
                if screenshot_result.success:
                    state.latest_screenshot_bytes = screenshot_result.data.get("screenshot_bytes")
                    if state.latest_screenshot_bytes:
                         logger.info(f"[{state.task_id}] Screenshot retrieved (bytes: {len(state.latest_screenshot_bytes)}).")
                    else:
                         logger.warning(f"[{state.task_id}] Screenshot tool succeeded but returned no bytes.")
                else:
                    logger.warning(f"[{state.task_id}] Failed to retrieve screenshot: {screenshot_result.error}")
            except AttributeError as ae:
                logger.error(f"[{state.task_id}] Error calling screenshot: {ae}", exc_info=True)
                state.latest_screenshot_bytes = None
                # Optionally set state error if tool method missing
            except Exception as e:
                logger.error(f"[{state.task_id}] Error getting Screenshot in prepare_context: {e}", exc_info=True)

        # Return empty dict as state (context, dom, screenshot) is modified directly
        # LangGraph will merge any returned dict into the state, but we modified state in-place.
        # If prepare_context itself encounters a critical error, returning {"error": ..., "final_outcome": ...}
        # allows the graph to potentially route to error handling or cleanup immediately.
        if state.error and state.final_outcome:
            return {"error": state.error, "final_outcome": state.final_outcome}
        else:
            return {}
        
    async def _generate_action(self, state: AgentState) -> Dict[str, Any]:
        """Generate the next action based on the current state and context."""
        logger.info(f"[{state.task_id}] Generating action...")
        
        # Context is now expected to be directly in state.context
        current_context = state.context

        # Prepare history for the prompt from AgentState.action_history
        history_entries = []
        for entry in state.action_history:
            # Safely handle None or missing keys
            action = entry.get("action", {})
            result = entry.get("result", {})
            action_str = json.dumps(action if action else {})
            result_str = json.dumps(result if result else {})
            history_entries.append(f"Action: {action_str}\\nResult: {result_str}")
        history_for_prompt = "\\n---\\n".join(history_entries)

        # Get available tool schemas from the registry
        try:
            available_tools_schema = self.tool_registry.get_tool_functions() # Use the correct method 
        except Exception as e:
             logger.error(f"[{state.task_id}] Failed to get tool functions from registry: {e}", exc_info=True)
             state.error = f"Failed to get tool functions: {e}"
             state.final_outcome = {"status": "error", "message": state.error}
             return {"error": state.error, "final_outcome": state.final_outcome} # Update state and return

        # --- Add Finish Tool Schema Explicitly --- 
        finish_tool_schema = {
            "finish": {
                "description": "Use this tool ONLY when you have the final answer to the user's original task and are ready to complete the task.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "output": {"type": "string", "description": "The final answer or result for the user's task."}
                    },
                    "required": ["output"]
                },
                "output_schema": {} # No output expected from this conceptual tool
            }
        }
        available_tools_schema.update(finish_tool_schema)
        logger.debug(f"[{state.task_id}] Added finish tool schema. Available tools: {list(available_tools_schema.keys())}")
        # --- End Add Finish Tool Schema ---

        logger.info(f"[{state.task_id}] >>> Calling LLM.generate_plan...") # Log Before Call
        try:
            plan_thought_confidence = await self.llm.generate_plan(
                task=state.task_description,
                context=current_context, # Use context directly from state
                history=history_for_prompt, # Pass formatted history string
                available_tools=available_tools_schema, # Pass schema
                dom_content=state.dom_content, # Pass DOM content from state
                screenshot_bytes=state.latest_screenshot_bytes # Pass screenshot from state
            )
            logger.info(f"[{state.task_id}] <<< LLM.generate_plan call finished.") # Log After Call
        except Exception as llm_error:
            logger.error(f"[{state.task_id}] <<< LLM.generate_plan call failed: {llm_error}", exc_info=True)
            # Handle the error appropriately - maybe set state error and return finish?
            error_msg = f"LLM generation failed: {llm_error}"
            plan = {"tool_name": "finish", "args": {"error": error_msg}}
            return {"error": error_msg, "current_action": plan} 

        thought, confidence, plan_steps = plan_thought_confidence
        
        # Assuming generate_plan returns exactly one step dict in plan_steps list
        if not plan_steps:
            error_msg = f"LLM did not generate any plan steps. Thought: {thought}"
            logger.error(f"[{state.task_id}] {error_msg}")
            # Handle error - maybe ask LLM to retry or terminate?
            # For now, set an error state and return a dummy finish action.
            # This error should likely be handled by the error routing mechanism.
            plan = {"tool_name": "finish", "args": {"error": error_msg}}
            # Return the error state to be merged by LangGraph
            return {"error": error_msg, "current_action": plan} 
        else:
             plan = plan_steps[0] # Take the first (and only) step
             # Adapt to tool_name/args used later (and in AgentState.current_action)
             plan = {"tool_name": plan.get('tool'), "args": plan.get('input')} 

        logger.info(f"[{state.task_id}] Generated action plan: {plan}")
        
        # --- Post-processing Step --- 
        # Explicitly check for and fix the literal '{{task}}' in web_search query
        if (
            plan.get("tool_name") == "web_search" and 
            isinstance(plan.get("args"), dict) and 
            plan["args"].get("query") == "{task}"
        ):
            logger.warning(f"[{state.task_id}] LLM generated literal '{{task}}' for web_search. Substituting with actual task description.")
            plan["args"]["query"] = state.task_description
            logger.info(f"[{state.task_id}] Corrected action plan: {plan}")
        # --- End Post-processing --- 

        # Return the potentially corrected plan under the 'current_action' key
        return {"current_action": plan}
        
    async def _execute_action(self, state: AgentState) -> Dict[str, Any]:
        """Execute the planned action from the state."""
        plan = state.current_action # Get action from state set by _generate_action
        if not plan or "tool_name" not in plan or plan["tool_name"] is None:
            error_msg = f"Invalid or missing action plan in state: {plan}"
            logger.error(f"[{state.task_id}] {error_msg}")
            # Return a ToolResult-like dict for consistency, which routing can check
            return {"result": {"success": False, "status": "error", "error": error_msg}}

        tool_name = plan["tool_name"]
        tool_args = plan.get("args", {})
        
        # Handle the conceptual 'finish' tool
        if tool_name == "finish":
             logger.info(f"[{state.task_id}] Received 'finish' action. Setting final outcome.")
             final_message = tool_args.get("result", "Task finished by agent.") # Note: LLM currently uses 'output'
             if "output" in tool_args and "result" not in tool_args:
                 final_message = tool_args["output"] # Use 'output' if 'result' isn't there

             if tool_args.get("error"): # If finish was called due to an error
                  final_status = "error"
                  error_detail = tool_args.get("error")
                  outcome = {"status": final_status, "message": final_message, "error": error_detail}
                  state.set_final_outcome(outcome) # Set internally for immediate use if needed
                  # Return outcome and error result for routing and state merging
                  return {"final_outcome": outcome, "result": {"success": False, "status": "error", "error": error_detail}}
             else:
                  final_status = "success"
                  outcome = {"status": final_status, "message": final_message}
                  state.set_final_outcome(outcome) # Set internally for immediate use if needed
                  # Return outcome and success result for routing and state merging
                  return {"final_outcome": outcome, "result": {"success": True, "status": "success", "message": final_message}}

        logger.info(f"[{state.task_id}] Executing action: {tool_name} with args: {tool_args}")
        
        try:
            # Execute using the registry's execute_tool method
            # This method should return a dict matching ToolResult structure
            result_dict = await self.tool_registry.execute_tool(tool_name, tool_args)
            
            # Log based on the success/status key within the result dictionary
            success = result_dict.get("success", False)
            status = result_dict.get("status", "error" if not success else "success")
            logger.info(f"[{state.task_id}] Action {tool_name} executed. Status: {status}")
            if status == "error":
                error_detail = result_dict.get('error', 'Unknown tool execution error')
                logger.warning(f"[{state.task_id}] Action {tool_name} failed. Error: {error_detail}")
            
            # IMPORTANT: Return the result under the "result" key for LangGraph state merging
            return {"result": result_dict}

        except Exception as e:
            error_msg = f"Critical error executing tool '{tool_name}': {e}"
            logger.error(f"[{state.task_id}] {error_msg}", exc_info=True)
            # Return a ToolResult-like dict for consistency
            return {"result": {"success": False, "status": "error", "error": error_msg}}
            
    async def _update_state(self, state: AgentState) -> Dict[str, Any]:
        """Update the agent state with the action result passed via state."""
        # Get the result dict from the state (populated by graph from execute_action output)
        action_result = state.result if state.result is not None else {}
        logger.info(f"[{state.task_id}] Updating state with result: {action_result}")
        
        # Add action and result to state history using AgentState method
        if state.current_action: # Should always have a current action if we reached here
            state.add_action(state.current_action, action_result)
        else:
            logger.error(f"[{state.task_id}] Cannot update state, current_action is missing.")
            # This indicates a potential logic error in the graph flow
            
        # Add to memory (Memory class handles context window) - Use synchronous save_context
        try:
            # Ensure current_action is not None before serializing
            input_data = {'input': json.dumps(state.current_action if state.current_action else {})}
            output_data = {'output': json.dumps(action_result)}
            self.memory.save_context(inputs=input_data, outputs=output_data) # Removed await
            logger.debug(f"[{state.task_id}] Saved action/result to memory.")
        except Exception as e:
            logger.error(f"[{state.task_id}] Failed to save context to memory in _update_state: {e}", exc_info=True)
            # Decide if this error should halt the process or just be logged
            # Potentially set state.error here if memory failure is critical

        # Clear the result from the state after processing to avoid re-processing
        state.result = None 

        return {} # Return empty dict as state is modified directly
            
    async def _check_termination_node(self, state: AgentState) -> Dict[str, Any]:
        """Node function: Check if the task should terminate or continue, update state if needed."""
        logger.info(f"[{state.task_id}] Checking termination conditions (node)...")
        
        # Check if the final outcome is set (should be merged by LangGraph now)
        if state.final_outcome is not None:
             logger.info(f"[{state.task_id}] Node: Final outcome already set ({state.final_outcome.get('status')}). Ready to terminate.")
             return {}
             
        # Check for maximum retries (based on consecutive failures)
        if state.consecutive_failures >= self.config.max_retries:
            logger.warning(f"[{state.task_id}] Node: Maximum retries ({self.config.max_retries}) reached. Setting final outcome.")
            if not state.final_outcome: # Avoid overwriting if somehow already set
                 state.set_final_outcome({
                     "status": "error", 
                     "message": f"Task failed after {state.consecutive_failures} consecutive errors.", 
                     "error": state.error # Use the last recorded error
                 })
            return {}
            
        # Check for maximum iterations
        if len(state.action_history) >= self.config.max_iterations:
            logger.warning(f"[{state.task_id}] Node: Maximum iterations ({self.config.max_iterations}) reached. Setting final outcome.")
            if not state.final_outcome:
                 state.set_final_outcome({
                     "status": "error", 
                     "message": f"Task failed after reaching max {self.config.max_iterations} iterations."
                 })
            return {}
        
        # If no termination condition met, just return empty dict
        logger.info(f"[{state.task_id}] Node: No termination condition met.")
        return {}

    def _route_after_termination_check(self, state: AgentState) -> str:
        """Routing function: Decide whether to finish or continue based on state."""
        logger.info(f"[{state.task_id}] Routing after termination check...")
        if state.final_outcome is not None:
             logger.info(f"[{state.task_id}] Routing: Final outcome is set. Routing to finish.")
             return "finish"
        elif state.consecutive_failures >= self.config.max_retries:
             logger.info(f"[{state.task_id}] Routing: Max retries reached. Routing to finish.")
             return "finish"
        elif len(state.action_history) >= self.config.max_iterations:
             logger.info(f"[{state.task_id}] Routing: Max iterations reached. Routing to finish.")
             return "finish"
        else:
             logger.info(f"[{state.task_id}] Routing: Task continuing.")
             return "continue"

class TaskAgent(BaseAgent):
    """Agent specialized for task execution."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        super().__init__(config, browser)
        # Add task-specific tools
        # Removed incorrect registration: self.tool_registry.register(ToolConfig(...))

class ResearchAgent(BaseAgent):
    """Agent specialized for research tasks."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        super().__init__(config, browser)
        # Add research-specific tools
        # Removed incorrect registration: self.tool_registry.register(ToolConfig(...))
        # Removed incorrect registration: self.tool_registry.register(ToolConfig(...))

class AnalysisAgent(BaseAgent):
    """Agent specialized for data analysis."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        super().__init__(config, browser)
        # Add analysis-specific tools
        # Removed incorrect registration: self.tool_registry.register(ToolConfig(...))
        # Removed incorrect registration: self.tool_registry.register(ToolConfig(...)) 