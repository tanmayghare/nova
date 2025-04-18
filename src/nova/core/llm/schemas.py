"""Pydantic models for structured LLM output."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    """A single step in an execution plan."""
    tool: str = Field(..., description="The name of the tool to use")
    input: Dict[str, Any] = Field(..., description="Input parameters for the tool")
    
class PlanGenerationOutput(BaseModel):
    """Structured output for plan generation."""
    plan: List[PlanStep] = Field(..., description="List of plan steps to execute")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the plan")
    thought_process: str = Field(..., description="Reasoning behind the plan")
    
class ActionOutput(BaseModel):
    """Structured output for action generation."""
    tool: str = Field(..., description="The name of the tool to use")
    input: Dict[str, Any] = Field(..., description="Input parameters for the tool")
    thought: str = Field(..., description="Reasoning behind the action")
    
class RecoveryPlanOutput(BaseModel):
    """Structured output for error recovery."""
    tool: str = Field(..., description="The name of the tool to use for recovery")
    input: Dict[str, Any] = Field(..., description="Input parameters for the recovery tool")
    reason: str = Field(..., description="Explanation of the recovery strategy")
    
class CommandInterpretationOutput(BaseModel):
    """Structured output for command interpretation."""
    goal: str = Field(..., description="The interpreted goal of the command")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities from the command")
    plan: List[PlanStep] = Field(default_factory=list, description="Initial plan for achieving the goal") 