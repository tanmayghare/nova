"""Plan-and-Execute Agent Configuration."""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PlanExecuteAgentConfig:
    """Configuration for Plan-and-Execute agent.
    
    Attributes:
        max_iterations: Maximum number of plan execution cycles
        verbose: Whether to print detailed execution logs
        handle_parsing_errors: Whether to handle tool parsing errors gracefully
        allowed_tools: List of tool names that the agent can use
        planner_prompt: Custom prompt for the planning phase
        executor_prompt: Custom prompt for the execution phase
        max_plan_length: Maximum number of steps in a plan
        allow_plan_modification: Whether to allow plan modification during execution
    """
    
    max_iterations: int = 5
    verbose: bool = True
    handle_parsing_errors: bool = True
    allowed_tools: Optional[List[str]] = None
    max_plan_length: int = 10
    allow_plan_modification: bool = True
    
    planner_prompt: str = """You are a planning agent that creates detailed plans for task execution.
    Break down the task into clear, actionable steps.
    Each step should be specific and executable.
    Consider dependencies between steps and potential failure points.
    """
    
    executor_prompt: str = """You are an execution agent that follows plans to complete tasks.
    Execute each step carefully and report the results.
    If a step fails, explain why and suggest alternatives.
    """ 