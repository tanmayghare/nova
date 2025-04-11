import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from nova.tools.base.tool import ToolResult
from nova.tools.base.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """A step in a tool chain."""
    tool_name: str
    input_data: Dict[str, Any]
    depends_on: Optional[List[str]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0


class ChainResult:
    """Result of a tool chain execution."""

    def __init__(self) -> None:
        """Initialize chain result."""
        self.steps: List[Dict[str, Any]] = []
        self.success: bool = True
        self.error: Optional[str] = None
        self.total_execution_time: float = 0.0

    def add_step_result(self, step: ChainStep, result: ToolResult) -> None:
        """Add a step result to the chain result.
        
        Args:
            step: The chain step that was executed
            result: The result of the step execution
        """
        self.steps.append({
            "tool_name": step.tool_name,
            "input": step.input_data,
            "output": result.data,
            "success": result.success,
            "error": result.error,
            "execution_time": result.execution_time,
        })
        
        if not result.success:
            self.success = False
            self.error = result.error

        self.total_execution_time += result.execution_time


class ToolChain:
    """Execute multiple tools in sequence with dependency management."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Initialize the tool chain.
        
        Args:
            registry: Tool registry to use for tool execution
        """
        self.registry = registry
        self.steps: List[ChainStep] = []

    def add_step(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        depends_on: Optional[List[str]] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        """Add a step to the chain.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for the tool
            depends_on: List of step names this step depends on
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds
        """
        step = ChainStep(
            tool_name=tool_name,
            input_data=input_data,
            depends_on=depends_on or [],
            max_retries=max_retries,
            timeout=timeout,
        )
        self.steps.append(step)

    async def execute_step(self, step: ChainStep) -> ToolResult:
        """Execute a single step with retry logic.
        
        Args:
            step: The step to execute
            
        Returns:
            Result of the step execution
        """
        while step.retry_count < step.max_retries:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.registry.execute_tool(step.tool_name, step.input_data),
                    timeout=step.timeout
                )
                
                if result.success:
                    return result
                
                # Retry on failure
                step.retry_count += 1
                if step.retry_count < step.max_retries:
                    logger.warning(
                        f"Step {step.tool_name} failed, retrying "
                        f"({step.retry_count}/{step.max_retries})"
                    )
                    await asyncio.sleep(1.0 * step.retry_count)  # Exponential backoff
                    continue
                
                return result
                
            except asyncio.TimeoutError:
                step.retry_count += 1
                error = f"Step {step.tool_name} timed out after {step.timeout} seconds"
                logger.error(error)
                
                if step.retry_count >= step.max_retries:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=error,
                        execution_time=step.timeout,
                    )
                    
            except Exception as e:
                step.retry_count += 1
                error = f"Error executing {step.tool_name}: {str(e)}"
                logger.error(error)
                
                if step.retry_count >= step.max_retries:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=error,
                        execution_time=0.0,
                    )
        
        return ToolResult(
            success=False,
            data=None,
            error=f"Step {step.tool_name} failed after {step.max_retries} attempts",
            execution_time=0.0,
        )

    def _get_ready_steps(
        self,
        completed_steps: Set[str],
        failed_steps: Set[str]
    ) -> List[ChainStep]:
        """Get steps that are ready to execute.
        
        Args:
            completed_steps: Set of completed step names
            failed_steps: Set of failed step names
            
        Returns:
            List of steps ready to execute
        """
        ready_steps = []
        
        for step in self.steps:
            # Skip completed and failed steps
            if step.tool_name in completed_steps or step.tool_name in failed_steps:
                continue
            
            # If step has no dependencies, it's ready
            if not step.depends_on:
                ready_steps.append(step)
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_steps for dep in step.depends_on):
                # Check if any dependencies failed
                if any(dep in failed_steps for dep in step.depends_on):
                    failed_steps.add(step.tool_name)
                    continue
                ready_steps.append(step)
        
        return ready_steps

    async def execute(self) -> ChainResult:
        """Execute all steps in the chain.
        
        Returns:
            Result of the chain execution
        """
        result = ChainResult()
        completed_steps: Set[str] = set()
        failed_steps: Set[str] = set()
        
        while len(completed_steps) + len(failed_steps) < len(self.steps):
            ready_steps = self._get_ready_steps(completed_steps, failed_steps)
            
            if not ready_steps:
                # No steps are ready but chain is not complete
                # This indicates a circular dependency or all remaining steps failed
                remaining = set(s.tool_name for s in self.steps) - completed_steps - failed_steps
                if remaining:
                    error = f"No steps ready to execute. Remaining steps: {remaining}"
                    logger.error(error)
                    result.success = False
                    result.error = error
                break
            
            # Execute ready steps in parallel
            tasks = [self.execute_step(step) for step in ready_steps]
            step_results = await asyncio.gather(*tasks)
            
            # Process results
            for step, step_result in zip(ready_steps, step_results):
                result.add_step_result(step, step_result)
                
                if step_result.success:
                    completed_steps.add(step.tool_name)
                else:
                    failed_steps.add(step.tool_name)
                    
                    # If step has dependents, mark them as failed
                    for other_step in self.steps:
                        if other_step.depends_on and step.tool_name in other_step.depends_on:
                            failed_steps.add(other_step.tool_name)
        
        return result 