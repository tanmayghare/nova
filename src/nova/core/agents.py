import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph

from .llm import LLM, LLMConfig
from .memory import Memory
from .tools import ToolRegistry, ToolConfig
from .browser import Browser
from ..tools.browser_tools import get_dom_impl

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
    browser: Optional[Browser] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    parallel_tasks: List[asyncio.Task] = field(default_factory=list)
    
    def add_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Add an action and its result to the history."""
        self.current_action = action
        self.action_history.append({
            "action": action,
            "result": result,
            "timestamp": str(uuid.uuid4())
        })
        
        if "error" in result and result.get("status") == "error":
            self.consecutive_failures += 1
            self.error = result["error"]
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
    memory_type: str = "buffer"
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
            memory_type=config.memory_type,
            summary_llm=self.llm.get_langchain_llm()
        )
        self.tool_registry = ToolRegistry()
        
        if config.tools:
            for tool_config in config.tools:
                self.tool_registry.register(tool_config)
                
        self._init_graph()
        
    def _init_graph(self) -> None:
        """Initialize the LangGraph state machine with enhanced error handling."""
        self.graph = StateGraph(AgentState)
        
        # Add nodes
        self.graph.add_node("prepare_context", self._prepare_context)
        self.graph.add_node("generate_action", self._generate_action)
        self.graph.add_node("execute_action", self._execute_action)
        self.graph.add_node("update_state", self._update_state)
        self.graph.add_node("check_termination", self._check_termination)
        self.graph.add_node("handle_error", self._handle_error)
        self.graph.add_node("cleanup", self._cleanup)
        
        # Add edges
        self.graph.add_edge("prepare_context", "generate_action")
        self.graph.add_edge("generate_action", "execute_action")
        self.graph.add_edge("execute_action", "update_state")
        self.graph.add_edge("update_state", "check_termination")
        
        # Add error handling edges
        self.graph.add_edge("execute_action", "handle_error", condition=self._has_error)
        self.graph.add_edge("handle_error", "generate_action", condition=self._can_recover)
        self.graph.add_edge("handle_error", "cleanup", condition=self._cannot_recover)
        
        # Add termination edges
        self.graph.add_conditional_edges(
            "check_termination",
            {
                "continue": "prepare_context",
                "finish": "cleanup"
            }
        )
        
        self.graph.set_entry_point("prepare_context")
        
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors and attempt recovery."""
        logger.error(f"Error in action execution: {state.error}")
        
        if state.recovery_attempts < state.max_recovery_attempts:
            # Generate recovery plan
            recovery_plan = await self.llm.generate_recovery_plan(
                error=state.error,
                action=state.current_action,
                dom_context=state.dom_content
            )
            
            if recovery_plan:
                state.current_action = recovery_plan
                state.recovery_attempts += 1
                logger.info(f"Recovery attempt {state.recovery_attempts}/{state.max_recovery_attempts}")
                return state
                
        # If we can't recover, mark for cleanup
        state.final_outcome = {
            "status": "error",
            "message": f"Failed after {state.recovery_attempts} recovery attempts",
            "error": state.error
        }
        return state
        
    async def _cleanup(self, state: AgentState) -> AgentState:
        """Clean up resources and finalize state."""
        try:
            # Clean up parallel tasks
            await state.cleanup_parallel_tasks()
            
            # Release browser if needed
            if state.browser:
                await state.browser.close()
                
            # Update memory with final state
            await self.memory.add(
                state.task_id,
                state.current_action,
                state.final_outcome
            )
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        return state
        
    def _has_error(self, state: AgentState) -> bool:
        """Check if the state has an error."""
        return state.error is not None
        
    def _can_recover(self, state: AgentState) -> bool:
        """Check if the error can be recovered from."""
        return (
            state.recovery_attempts < state.max_recovery_attempts and
            state.error is not None and
            state.final_outcome is None
        )
        
    def _cannot_recover(self, state: AgentState) -> bool:
        """Check if the error cannot be recovered from."""
        return not self._can_recover(state)
        
    async def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task with enhanced error handling."""
        logger.info(f"Starting agent run for task: {task}")
        
        state = AgentState(
            task_id=str(uuid.uuid4()),
            task_description=task,
            browser=self.browser
        )
        
        try:
            final_state = await self.graph.arun(state)
            
            if final_state.final_outcome and final_state.final_outcome.get("status") != "error":
                return {
                    "status": "success",
                    "result": final_state.final_outcome.get("message", "Task completed."),
                    "action_history": final_state.action_history
                }
            else:
                error_message = final_state.final_outcome.get("message", "Agent failed unexpectedly.")
                if final_state.error:
                    error_message = final_state.error
                    
                return {
                    "status": "error",
                    "error": error_message,
                    "action_history": final_state.action_history
                }
                
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "action_history": state.action_history
            }
            
        finally:
            # Ensure cleanup happens even if there's an error
            await self._cleanup(state)

    async def _prepare_context(self, state: AgentState) -> Dict[str, Any]:
        """Prepare context for action generation."""
        logger.info("--- Preparing Context ---")
        context = await self.memory.get_context(state.task_description)
        
        # Get current DOM state
        dom_result = await get_dom_impl(self.browser)
        dom_content = dom_result.get("dom") if dom_result.get("status") == "success" else None
        if not dom_content:
            logger.warning("Failed to retrieve DOM content.")
            
        return {
            "context": context,
            "dom_content": dom_content,
            "action_history": state.action_history
        }
        
    async def _generate_action(self, state: AgentState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the next action."""
        logger.info("--- Generating Action ---")
        
        # Check if we are recovering from an error
        if state.get('error') and state.get('current_action'):
            logger.warning(f"Generating recovery plan for error: {state['error']}")
            plan = await self.llm.generate_recovery_plan(
                error=state['error'],
                action=state['current_action'],
                dom_context=state['dom_content']
            )
            # Reset error state after generating recovery plan
            state['error'] = None 
            state['consecutive_failures'] = 0
        else:
            # Generate normal plan
            plan = await self.llm.generate_plan(
                task=state.task_description,
                context=context["context"],
                tools=context["tools"],
                action_history=context["action_history"],
                dom_context=context["dom_content"]
            )
            
        logger.debug(f"Generated plan: {plan}")
        return {"plan": plan}
        
    async def _execute_action(self, state: AgentState, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generated action."""
        logger.info("--- Executing Action ---")
        try:
            result = await self.tool_registry.execute_tool(
                plan["tool"],
                plan["input"]
            )
            logger.info(f"Tool {plan['tool']} execution result: {result}")
            return {"result": result}
        except Exception as e:
            logger.error(f"Action execution failed for tool {plan['tool']}: {e}", exc_info=True)
            return {"error": str(e)}
            
    async def _update_state(self, state: AgentState, result: Dict[str, Any]) -> Dict[str, Any]:
        """Update the agent state."""
        logger.info("--- Updating State ---")
        if "error" in result:
            # Generate recovery plan
            recovery_plan = await self.llm.generate_recovery_plan(
                error=result["error"],
                action=state.current_action,
                dom_context=state.dom_content
            )
            return {"recovery_plan": recovery_plan}
        else:
            # Add successful action to history
            await self.memory.add(
                state.task_id,
                state.current_action,
                result["result"]
            )
            return {"success": True}
            
    async def _check_termination(self, state: AgentState, update: Dict[str, Any]) -> str:
        """Check if the agent should terminate."""
        logger.info("--- Checking Termination ---")
        if "error" in update and state.consecutive_failures >= self.config.max_retries:
            logger.error(f"Termination condition: Max retries ({self.config.max_retries}) exceeded.")
            state.final_outcome = {"status": "error", "message": f"Agent failed after {self.config.max_retries} consecutive errors."}
            return "finish"
        if len(state.action_history) >= self.config.max_iterations:
            logger.warning(f"Termination condition: Max iterations ({self.config.max_iterations}) reached.")
            state.final_outcome = {"status": "error", "message": f"Agent stopped after {self.config.max_iterations} iterations."}
            return "finish"
        return "continue"

class TaskAgent(BaseAgent):
    """Agent specialized for task execution."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        super().__init__(config, browser)
        # Add task-specific tools
        self.tool_registry.register(ToolConfig(
            name="execute_task",
            description="Execute a task step",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "parameters": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "result": {"type": "object"}
                }
            }
        ))

class ResearchAgent(BaseAgent):
    """Agent specialized for research tasks."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        super().__init__(config, browser)
        # Add research-specific tools
        self.tool_registry.register(ToolConfig(
            name="search_web",
            description="Search the web for information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array", "items": {"type": "object"}}
                }
            }
        ))
        self.tool_registry.register(ToolConfig(
            name="analyze_content",
            description="Analyze content for key information",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "focus": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}}
                }
            }
        ))

class AnalysisAgent(BaseAgent):
    """Agent specialized for data analysis."""
    
    def __init__(self, config: AgentConfig, browser: Browser):
        super().__init__(config, browser)
        # Add analysis-specific tools
        self.tool_registry.register(ToolConfig(
            name="analyze_data",
            description="Analyze data using statistical methods",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}},
                    "method": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "statistics": {"type": "object"},
                    "visualization": {"type": "string"}
                }
            }
        ))
        self.tool_registry.register(ToolConfig(
            name="generate_report",
            description="Generate an analysis report",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "format": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "report": {"type": "string"},
                    "format": {"type": "string"}
                }
            }
        )) 